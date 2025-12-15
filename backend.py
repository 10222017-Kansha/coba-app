# backend.py
import json
import os
import time
import requests
import subprocess
import gdown
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity




# API KEYS (Disimpan di backend)
API_KEY_ASSEMBLYAI = "4a4823f3f6df4259b8c9d5e3cd8e545c"
model = SentenceTransformer('all-MiniLM-L6-v2') 
# GOOGLE_API_KEY = "AIzaSyB6-Fk2cbzeFcqV142B1E6i5OP5qS7pFiE"

def parse_input_json(json_data):
    video_links = {}
    try:
        interviews = json_data['data']['reviewChecklists']['interviews']
        for item in interviews:
            vid_url = item.get('recordedVideoUrl', '')
            pos_id = item.get('positionId')
            if vid_url and ("drive.google.com" in vid_url or "youtube" in vid_url):
                video_links[pos_id] = vid_url
    except Exception:
        return {}
    return video_links

def process_videos_pipeline(links_dict):
    output_audio_dir = "temp_audios"
    output_video_dir = "temp_videos_mute"
    os.makedirs(output_audio_dir, exist_ok=True)
    os.makedirs(output_video_dir, exist_ok=True)
    
    audio_paths = []
    video_paths = []
    sorted_ids = sorted(links_dict.keys())
    
    for key in sorted_ids:
        url = links_dict[key]
        try:
            if "drive.google.com" in url:
                file_id = url.split('/d/')[1].split('/')[0]
                temp_mp4 = f"temp_raw_{key}.mp4"
                gdown.download(id=file_id, output=temp_mp4, quiet=True, fuzzy=True)
            else:
                audio_paths.append(None); video_paths.append(None)
                continue

            if os.path.exists(temp_mp4):
                # Extract Audio
                audio_out = os.path.join(output_audio_dir, f"audio_{key}.mp3")
                subprocess.run(['ffmpeg', '-y', '-i', temp_mp4, '-vn', '-acodec', 'libmp3lame', '-ab', '192k', audio_out], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                audio_paths.append(os.path.abspath(audio_out))
                
                # Mute Video
                video_out = os.path.join(output_video_dir, f"video_{key}_mute.mp4")
                subprocess.run(['ffmpeg', '-y', '-i', temp_mp4, '-c', 'copy', '-an', video_out], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                video_paths.append(os.path.abspath(video_out))
                
                os.remove(temp_mp4)
            else:
                audio_paths.append(None); video_paths.append(None)
        except Exception:
            audio_paths.append(None); video_paths.append(None)

    return audio_paths, video_paths

def transcribe_audios(audio_filepaths):
    headers = {'authorization': API_KEY_ASSEMBLYAI}
    transcriptions = []
    
    for path in audio_filepaths:
        if path is None:
            transcriptions.append("")
            continue
            
        def read_file(fname):
            with open(fname, 'rb') as f:
                while True:
                    data = f.read(5242880)
                    if not data: break
                    yield data
        try:
            upload_resp = requests.post('https://api.assemblyai.com/v2/upload', headers=headers, data=read_file(path))
            upload_url = upload_resp.json()['upload_url']
            ts_resp = requests.post('https://api.assemblyai.com/v2/transcript', json={'audio_url': upload_url}, headers=headers)
            ts_id = ts_resp.json()['id']
            while True:
                poll_resp = requests.get(f'https://api.assemblyai.com/v2/transcript/{ts_id}', headers=headers)
                status = poll_resp.json()['status']
                if status == 'completed': transcriptions.append(poll_resp.json()['text']); break
                elif status == 'error': transcriptions.append(""); break
                time.sleep(2)
        except:
            transcriptions.append("")
    return transcriptions



def grade_answers(transcript_list, rubric_json_path):
    """
    Evaluate answers based on vector similarity (Semantic Search) 
    without using external APIs.
    """
    
    # Load Rubrik
    with open(rubric_json_path, 'r') as f:
        rubric_data = json.load(f)
    
    scores_list = []
    reasons_list = []
    
    print(f"\nStarting assessment for {len(transcript_list)} answers...")

    for index, transcript_text in enumerate(transcript_list):
        if index >= len(rubric_data): break
            
        current_rubric = rubric_data[index]
        rubric_criteria = current_rubric['rubric'] 
        
        rubric_texts = []
        rubric_scores_key = []
        
        for score_key, description in rubric_criteria.items():
            # Skip if description is empty/None
            if description:
                rubric_texts.append(description)
                rubric_scores_key.append(int(score_key))
        
        if not rubric_texts:
            scores_list.append(0)
            reasons_list.append("Tidak ada deskripsi rubrik.")
            continue

        # Tokenization
        student_embedding = model.encode([transcript_text]) 
        
        # Vector description section
        rubric_embeddings = model.encode(rubric_texts)
        
        # Calculate similarity with cosine similarity
        # The result is an array of similarities between student answers and [desc_4, desc_3, desc_2, desc_1]
        similarities = cosine_similarity(student_embedding, rubric_embeddings)[0]
        
        # Find the highest value (Which one is most similar to the description?)
        best_match_index = np.argmax(similarities)
        best_score = rubric_scores_key[best_match_index]
        similarity_percent = similarities[best_match_index] * 100
        
        # If the similarity is very low (e.g., < 15%), consider the answer irrelevant (Score 0 or 1)
        if similarity_percent < 15:
            final_score = 1
            final_reason = f"Answer {current_rubric['id']} was found to be irrelevant to any of the rubric criteria."
        else:
            final_score = best_score
            matched_desc = rubric_texts[best_match_index]
            final_reason = f"{matched_desc[:100]}."

        print(f"Answer {current_rubric['id']} has been evaluated")

        scores_list.append(final_score)
        reasons_list.append(final_reason)

    return scores_list, reasons_list

def generate_final_report_v2(input_payload, transcripts, scores, reasons):
    try:
        past_data = input_payload['data']['pastReviews'][0]
        assessor = past_data.get('assessorProfile', {})
        decision = past_data.get('decision', 'Need Human')
        reviewed_at = past_data.get('reviewedAt', '')
        proj_score = past_data.get('scoresOverview', {}).get('project', 100)
    except: assessor, decision, reviewed_at, proj_score = {}, "Need Human", "", 100
        
    sum_score = sum(scores)
    max_score = 4 * len(scores) if scores else 1
    interview_val = (sum_score / max_score) * 100
    total_val = (0.7 * proj_score) + (0.3 * interview_val)
    avg = sum_score / len(scores) if scores else 0
    overall_notes = f"Average Score: {avg:.1f}. "
    if avg >= 3.0: overall_notes += "Candidate demonstrates strong understanding."
    elif avg >= 2.0: overall_notes += "Candidate shows basic understanding."
    else: overall_notes += "Candidate answers are insufficient."
    
    det_scores = []
    try:
        qs = input_payload['data']['reviewChecklists']['interviews']
        qs = sorted(qs, key=lambda x: x['positionId'])
    except: qs = []
        
    for i in range(len(scores)):
        pid = qs[i]['positionId'] if i < len(qs) else i+1
        det_scores.append({
            "id": pid, "score": scores[i], "reason": reasons[i],
            "transcriptions": transcripts[i]
        })
        
    final = {
        "assessorProfile": assessor, "decision": decision, "reviewedAt": reviewed_at,
        "scoresOverview": {"project": proj_score, "interview": round(interview_val, 1), "total": round(total_val, 1)},
        "reviewChecklistResult": {"project": [], "interviews": {"minScore": 0, "maxScore": 4, "scores": det_scores}},
        "Overall notes": overall_notes
    }

    return final
