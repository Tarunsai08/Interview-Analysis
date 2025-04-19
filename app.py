import os
import io
import time
import tempfile
import logging
import base64
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from flask_cors import CORS

import cv2
import numpy as np
import mediapipe as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from deepface import DeepFace
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import threading
import multiprocessing as mpr

from pydub import AudioSegment, silence
from pydub.playback import play
import speech_recognition as sr
from textblob import TextBlob
import spacy
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Spacy model
nlp = spacy.load("en_core_web_sm")

# NLTK download
nltk.download('punkt')
nltk.download('stopwords')

# List of filler words
filler_words = ['um', 'uh', 'like', 'you know', 'basically', 'actually', 'seriously']

class InterviewAnalyzer:
    """
    A class to analyze interview scenarios including response latency, head pose estimation, and speech rate analysis.
    """
    def __init__(self):
        self.response_times = []
        self.head_pose_estimations = []
        self.speech_rates = []
        self.last_question_time = None
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_3d = []
        self.face_2d = []

    def start_question_timer(self):
        """Start the timer when a question is asked."""
        self.last_question_time = time.time()

    def record_response_time(self):
        """Record the response time when an answer is given."""
        if self.last_question_time is not None:
            response_time = time.time() - self.last_question_time
            self.response_times.append(response_time)
            self.last_question_time = None

    def estimate_head_pose(self, frame: np.ndarray) -> Tuple[float, float, float]:
        """
        Estimate the head pose in a single frame.
        
        Args:
            frame (np.ndarray): Input frame in RGB format
        
        Returns:
            Tuple containing the yaw, pitch, and roll angles
        """
        height, width = frame.shape[:2]
        results = self.face_mesh.process(frame)
        
        if not results.multi_face_landmarks:
            return 0.0, 0.0, 0.0
        
        face_landmarks = results.multi_face_landmarks[0]
        self.face_3d = []
        self.face_2d = []
        
        for idx, lm in enumerate(face_landmarks.landmark):
            if idx in [33, 263, 1, 61, 291, 199]:
                x, y = int(lm.x * width), int(lm.y * height)
                self.face_2d.append([x, y])
                self.face_3d.append([x, y, lm.z])
        
        face_2d_np = np.array(self.face_2d, dtype=np.float64)
        face_3d_np = np.array(self.face_3d, dtype=np.float64)
        
        focal_length = 1 * width
        camera_matrix = np.array([[focal_length, 0, height / 2],
                                  [0, focal_length, width / 2],
                                  [0, 0, 1]])
        
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d_np, face_2d_np, camera_matrix, dist_matrix)
        
        rot_matrix, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_matrix)
        
        yaw, pitch, roll = [angle * 360 for angle in angles]
        self.head_pose_estimations.append((yaw, pitch, roll))
        
        return yaw, pitch, roll

    def analyze_speech_rate(self, transcribed_text: str, duration_seconds: float) -> float:
        """
        Analyze the speech rate (words per minute) of the transcribed text.

        Args:
            transcribed_text (str): The transcribed text.
            duration_seconds (float): The total duration of the speech in seconds.

        Returns:
            float: The speech rate in words per minute.
        """
        if duration_seconds <= 0:
            return 0  # Prevent division by zero

        words = word_tokenize(transcribed_text)
        num_words = len(words)
        duration_minutes = duration_seconds / 60  # Convert seconds to minutes
        
        speech_rate = num_words / duration_minutes  # Words per minute (WPM)
        self.speech_rates.append(speech_rate)

        return speech_rate

    def get_analysis_results(self) -> Dict[str, any]:
        """Get the results of the interview analysis."""
        return {
            'response_times': self.response_times,
            'head_pose_estimations': self.head_pose_estimations,
            'speech_rates': self.speech_rates,
        }

class GazeDetector:
    """
    A class to detect and analyze gaze direction using MediaPipe Face Mesh.
    Detects if user is looking left, right, up, down, or straight at camera.
    """
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark indices for eyes
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
        # Threshold for determining straight gaze (in pixels)
        self.STRAIGHT_THRESHOLD = 1
        
    def analyze_gaze(self, frame: np.ndarray) -> Dict[str, any]:
        """
        Analyze gaze direction in a single frame.
        
        Args:
            frame (np.ndarray): Input frame in RGB format
            
        Returns:
            Dict containing gaze analysis results including direction and eye aspect ratio
        """
        height, width = frame.shape[:2]
        results = self.face_mesh.process(frame)
        
        if not results.multi_face_landmarks:
            return None
            
        face_landmarks = results.multi_face_landmarks[0]
        
        # Get iris and eye landmarks
        mesh_points = np.array([
            np.multiply([p.x, p.y], [width, height]).astype(int)
            for p in face_landmarks.landmark
        ])
        
        left_iris = np.mean(mesh_points[self.LEFT_IRIS], axis=0).astype(int)
        right_iris = np.mean(mesh_points[self.RIGHT_IRIS], axis=0).astype(int)
        
        # Calculate relative positions of irises within eyes
        left_eye_points = mesh_points[self.LEFT_EYE]
        right_eye_points = mesh_points[self.RIGHT_EYE]
        
        # Calculate eye centers
        left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
        right_eye_center = np.mean(right_eye_points, axis=0).astype(int)
        
        # Calculate displacement vectors for both eyes
        left_displacement = left_iris - left_eye_center
        right_displacement = right_iris - right_eye_center
        
        # Average displacement across both eyes
        avg_dx = (left_displacement[0] + right_displacement[0]) / 2
        avg_dy = (left_displacement[1] + right_displacement[1]) / 2
        
        # Determine gaze direction using displacement and threshold
        if abs(avg_dx) <= self.STRAIGHT_THRESHOLD and abs(avg_dy) <= self.STRAIGHT_THRESHOLD:
            final_direction = 'straight'
        elif abs(avg_dx) > abs(avg_dy):
            final_direction = 'left' if avg_dx < 0 else 'right'
        else:
            final_direction = 'up' if avg_dy < 0 else 'down'
            
        # Calculate confidence metric based on displacement magnitude
        displacement_magnitude = np.sqrt(avg_dx**2 + avg_dy**2)
        max_possible_displacement = np.sqrt((width/4)**2 + (height/4)**2)  # Theoretical maximum
        direction_confidence = 1.0 - min(displacement_magnitude / max_possible_displacement, 1.0)
        
        return {
            'direction': final_direction,
            'left_iris': left_iris.tolist(),
            'right_iris': right_iris.tolist(),
            'displacement': {
                'x': float(avg_dx),
                'y': float(avg_dy)
            },
            'confidence': direction_confidence
        }

class AttentionDetector:
    """
    Detects and analyzes attention levels based on facial landmarks and behavior patterns.
    """
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmarks for attention detection
        self.BLINK_LANDMARKS = [386, 374, 373, 390, 159, 145, 144, 163]
        self.HEAD_LANDMARKS = [33, 133, 362, 263]
        
        # Thresholds
        self.BLINK_THRESHOLD = 0.3
        self.HEAD_MOVEMENT_THRESHOLD = 0.1
        self.ATTENTION_WINDOW = 30  # frames
        
    def detect_blinks(self, landmarks, frame_shape) -> float:
        """Calculate eye openness ratio"""
        if not landmarks:
            return 0.0
            
        height, width = frame_shape[:2]
        points = []
        
        for idx in self.BLINK_LANDMARKS:
            point = landmarks.landmark[idx]
            points.append([int(point.x * width), int(point.y * height)])
            
        # Calculate eye aspect ratio
        left_eye = points[:4]
        right_eye = points[4:]
        
        left_ear = self._eye_aspect_ratio(left_eye)
        right_ear = self._eye_aspect_ratio(right_eye)
        
        return (left_ear + right_ear) / 2
        
    def _eye_aspect_ratio(self, eye_points) -> float:
        """Calculate eye aspect ratio from points"""
        vertical_1 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[3]))
        vertical_2 = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[2]))
        horizontal = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
        
        if horizontal == 0:
            return 0.0
            
        return (vertical_1 + vertical_2) / (2.0 * horizontal)
        
    def detect_attention(self, frame: np.ndarray) -> Dict[str, any]:
        """
        Detect attention level in a single frame
        
        Args:
            frame (np.ndarray): Input frame in RGB format
            
        Returns:
            Dict containing attention metrics
        """
        results = self.face_mesh.process(frame)
        
        if not results.multi_face_landmarks:
            return {
                'attention_level': 0.0,
                'blink_rate': 0.0,
                'head_movement': 0.0,
                'is_attentive': False
            }
            
        landmarks = results.multi_face_landmarks[0]
        
        # Calculate metrics
        blink_ratio = self.detect_blinks(landmarks, frame.shape)
        head_movement = self._calculate_head_movement(landmarks, frame.shape)
        
        # Determine attention level
        attention_level = self._calculate_attention_level(blink_ratio, head_movement)
        
        return {
            'attention_level': attention_level,
            'blink_rate': blink_ratio,
            'head_movement': head_movement,
            'is_attentive': attention_level > 0.6
        }
        
    def _calculate_head_movement(self, landmarks, frame_shape) -> float:
        """Calculate head movement from landmarks"""
        height, width = frame_shape[:2]
        points = []
        
        for idx in self.HEAD_LANDMARKS:
            point = landmarks.landmark[idx]
            points.append([point.x * width, point.y * height])
            
        # Calculate movement as average displacement from center
        center = np.mean(points, axis=0)
        movements = [np.linalg.norm(np.array(p) - center) for p in points]
        return np.mean(movements) / width  # Normalize by frame width
        
    def _calculate_attention_level(self, blink_ratio: float, head_movement: float) -> float:
        """Calculate overall attention level"""
        blink_score = 1.0 if blink_ratio > self.BLINK_THRESHOLD else 0.5
        movement_score = 1.0 if head_movement < self.HEAD_MOVEMENT_THRESHOLD else 0.5
        
        return (blink_score * 0.4 + movement_score * 0.6)  # Weighted average

class EmotionGazeAnalyzer:
    """
    A class to analyze emotions, gaze, and attention in video content.
    """
    def __init__(self, video_path: str, sample_interval: float = 1.0):
        self.video_path = video_path
        self.sample_interval = sample_interval
        self.emotion_counts = defaultdict(int)
        self.gaze_counts = defaultdict(int)
        self.attention_scores = []
        self.total_frames_processed = 0
        self.gaze_detector = GazeDetector()
        self.attention_detector = AttentionDetector()
        
    def load_video(self) -> Tuple[cv2.VideoCapture, float]:
        """Load the video file and get its properties."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video file {self.video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        return cap, fps
        
    def analyze_frame(self, frame: np.ndarray) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Analyze a single frame to detect emotion and gaze direction.
        
        Args:
            frame (np.ndarray): Input frame in RGB format
            
        Returns:
            Tuple containing emotion and gaze analysis results
        """
        try:
            # Emotion analysis
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            emotions = result[0]['emotion'] if isinstance(result, list) else result['emotion']
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            
            # Gaze analysis
            gaze_result = self.gaze_detector.analyze_gaze(frame)
            
            return dominant_emotion, gaze_result
            
        except Exception as e:
            logger.warning(f"Error processing frame: {str(e)}")
            return None, None
    
    def process_video(self) -> Dict[str, any]:
        """Process the video file and analyze emotions, gaze, and attention."""
        cap, fps = self.load_video()
        frames_to_skip = max(1, int(fps * self.sample_interval))
        frame_count = 0
        
        logger.info(f"Starting video processing... FPS: {fps}, Processing every {frames_to_skip} frames")
        start_time = time.time()
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frames_to_skip == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Analyze emotion and gaze
                    emotion, gaze = self.analyze_frame(frame_rgb)
                    
                    # Analyze attention
                    attention = self.attention_detector.detect_attention(frame_rgb)
                    
                    if emotion and gaze and attention:
                        self.emotion_counts[emotion] += 1
                        self.gaze_counts[gaze['direction']] += 1
                        self.attention_scores.append(attention['attention_level'])
                        self.total_frames_processed += 1
                        logger.info(f"Processed frame {frame_count}: Emotion - {emotion}, Gaze - {gaze['direction']}, Attention - {attention['attention_level']}")
                
                frame_count += 1
                
        except Exception as e:
            logger.error(f"Error during video processing: {str(e)}")
            raise
        finally:
            cap.release()
        
        processing_time = time.time() - start_time
        logger.info(f"Video processing completed in {processing_time:.2f} seconds")
        
        return {
            'emotions': self.calculate_emotion_distribution(),
            'gaze': self.calculate_gaze_distribution(),
            'attention': self.calculate_attention_metrics(),
            'metrics': self.calculate_engagement_metrics()
        }
    
    def calculate_emotion_distribution(self) -> Dict[str, float]:
        """Calculate the distribution of emotions as percentages."""
        if self.total_frames_processed == 0:
            return {emotion: 0.0 for emotion in 
                   ['neutral', 'happy', 'sad', 'angry', 'fear', 'surprise', 'disgust']}
        
        distribution = {
            emotion: (count / self.total_frames_processed) * 100
            for emotion, count in self.emotion_counts.items()
        }
        
        # Ensure all emotions are present
        all_emotions = ['neutral', 'happy', 'sad', 'angry', 'fear', 'surprise', 'disgust']
        for emotion in all_emotions:
            if emotion not in distribution:
                distribution[emotion] = 0.0
                
        return distribution
    
    def calculate_gaze_distribution(self) -> Dict[str, float]:
        """Calculate the distribution of gaze directions as percentages."""
        if self.total_frames_processed == 0:
            return {direction: 0.0 for direction in ['up', 'down', 'left', 'right', 'straight']}
            
        return {
            direction: (count / self.total_frames_processed) * 100
            for direction, count in self.gaze_counts.items()
        }
    
    def calculate_attention_metrics(self) -> Dict[str, float]:
        """Calculate attention metrics from collected scores"""
        if not self.attention_scores:
            return {
                'average_attention': 0.0,
                'attention_drops': 0,
                'sustained_attention_periods': 0
            }
            
        attention_array = np.array(self.attention_scores)
        
        return {
            'average_attention': float(np.mean(attention_array)),
            'attention_drops': int(np.sum(attention_array < 0.3)),
            'sustained_attention_periods': int(np.sum(attention_array > 0.7))
        }
        
    def calculate_engagement_metrics(self) -> Dict[str, float]:
        """
        Calculate engagement metrics based on emotion, gaze, and attention patterns.
        """
        if self.total_frames_processed == 0:
            return {'engagement': 0.0, 'confidence': 0.0, 'nervousness': 0.0}
        
        # Calculate confidence and nervousness as before
        confidence_score = 0
        nervousness_score = 0
        
        if self.emotion_counts['happy'] + self.emotion_counts['neutral'] > \
           self.emotion_counts['fear'] + self.emotion_counts['sad']:
            confidence_score = 75 + (self.emotion_counts['happy'] / self.total_frames_processed) * 100
        else:
            confidence_score = 50
        
        nervous_emotions = ['fear', 'surprise', 'sad']
        nervousness_score = sum(self.emotion_counts[emotion] 
                              for emotion in nervous_emotions) / self.total_frames_processed * 100
        
        # Calculate engagement based on both emotion, gaze, and attention
        forward_gaze = self.gaze_counts['left'] + self.gaze_counts['right']  # Assuming side-to-side is more engaged
        positive_emotions = self.emotion_counts['happy'] + self.emotion_counts['surprise']
        avg_attention = np.mean(self.attention_scores) if self.attention_scores else 0
        
        engagement_score = (
            (forward_gaze / self.total_frames_processed) * 0.4 +  # Weight gaze
            (positive_emotions / self.total_frames_processed) * 0.4 +  # Weight emotions
            avg_attention * 0.2  # Weight attention
        ) * 100
        
        return {
            'confidence': min(100, confidence_score),
            'nervousness': min(100, nervousness_score),
            'engagement': min(100, engagement_score)
        }

class VideoAnalyzer:
    """
    Integrated class for video, emotion, gaze, sentiment, and attention analysis
    """
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.emotion_gaze_analyzer = EmotionGazeAnalyzer(video_path)
        self.interview_analyzer = InterviewAnalyzer()
        
    def extract_audio(self) -> str:
        """Extract audio from video file"""
        try:
            audio_output_path = os.path.join(tempfile.gettempdir(), "temp_audio.wav")
            audio = AudioSegment.from_file(self.video_path)
            audio.export(audio_output_path, format="wav")
            return audio_output_path
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            raise

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio to text"""
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            logger.warning("Speech recognition could not understand audio")
            return ""
        except sr.RequestError as e:
            logger.error(f"Could not request results from speech recognition service: {str(e)}")
            return ""

    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of transcribed text"""
        if not text:
            return {
                'polarity': 0,
                'subjectivity': 0,
                'conclusion': 'Neutral',
                'filler_words': 0,
                'keywords': []
            }

        blob = TextBlob(text)
        sentiment = blob.sentiment
        
        # Get keywords
        doc = nlp(text)
        keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
        
        # Count filler words
        tokens = word_tokenize(text.lower())
        filler_count = sum(1 for word in tokens if word in filler_words)
        
        return {
            'polarity': sentiment.polarity,
            'subjectivity': sentiment.subjectivity,
            'conclusion': 'Positive' if sentiment.polarity > 0 else 'Negative' if sentiment.polarity < 0 else 'Neutral',
            'filler_words': filler_count,
            'keywords': keywords[:10]  # Top 10 keywords
        }

    def analyze_audio_patterns(self, audio_path: str) -> Dict:
        """Analyze audio patterns including silence"""
        audio = AudioSegment.from_file(audio_path)
        silence_threshold = -40  # dBFS
        min_silence_len = 2000  # 1 second
        
        silent_segments = silence.detect_silence(
            audio, 
            min_silence_len=min_silence_len,
            silence_thresh=silence_threshold
        )

        audio_duration = len(audio)/1000 
        
        total_silence_duration = sum((end - start) for start, end in silent_segments) / 1000
        
        return {
            'total_silence_duration': total_silence_duration,
            'silence_segments_count': len(silent_segments),
            'audio_duration': audio_duration
        }

    def generate_waveform(self, audio_path: str) -> str:
        """Generate and save audio waveform visualization"""
        audio = AudioSegment.from_file(audio_path)
        samples = np.array(audio.get_array_of_samples())
        
        plt.figure(figsize=(10, 4))
        plt.plot(samples, color='blue')
        plt.title('Audio Waveform')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        return base64.b64encode(buf.getvalue()).decode()

    def analyze(self) -> Dict:
        try:
            # Emotion, gaze, and attention analysis
            analysis_results = self.emotion_gaze_analyzer.process_video()
            
            # Audio analysis
            audio_path = self.extract_audio()
            transcribed_text = self.transcribe_audio(audio_path)
            sentiment_results = self.analyze_sentiment(transcribed_text)
            audio_patterns = self.analyze_audio_patterns(audio_path)
            waveform = self.generate_waveform(audio_path)
            
            # Start and record response times as appropriate
            self.interview_analyzer.start_question_timer()
            self.interview_analyzer.record_response_time()
            
            # Head pose estimation example
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            if ret:
                yaw, pitch, roll = self.interview_analyzer.estimate_head_pose(frame)
            
            # Speech rate analysis
            speech_rate = self.interview_analyzer.analyze_speech_rate(transcribed_text,audio_patterns['audio_duration'])
            
            # Get interview analysis results
            interview_results = self.interview_analyzer.get_analysis_results()
            
            # Calculate overall engagement
            engagement_score = self._calculate_overall_engagement(
                analysis_results,
                sentiment_results,
                audio_patterns
            )
            
            # Clean up
            os.remove(audio_path)
            
            return {
                'success': True,
                'emotions': analysis_results['emotions'],
                'gaze': analysis_results['gaze'],
                'attention': analysis_results['attention'],
                'metrics': analysis_results['metrics'],
                'engagement': engagement_score,
                'sentiment': sentiment_results,
                'audio_patterns': audio_patterns,
                'transcribed_text': transcribed_text,
                # 'waveform': waveform,
                'interview_analysis': interview_results,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            logger.error(f"Error in complete analysis: {str(e)}")
            raise     

    def _calculate_overall_engagement(self, 
                                    analysis_results: Dict,
                                    sentiment_results: Dict,
                                    audio_patterns: Dict) -> float:
        """Calculate overall engagement score considering all factors"""
        # Attention weight (35%)
        attention_score = analysis_results['attention']['average_attention'] * 100
        
        # Emotion and gaze weight (25%)
        behavior_score = analysis_results['metrics']['confidence']
        
        # Sentiment weight (20%)
        sentiment_score = (
            (sentiment_results['polarity'] + 1) / 2 * 60 +  # Normalize to 0-60
            (1 - sentiment_results['filler_words'] / 100) * 40  # Penalize filler words
        )
        
        # Audio patterns weight (20%)
        total_duration = audio_patterns['total_silence_duration']
        audio_score = max(0, 100 - (total_duration / 60))  # Penalize excessive silence
        
        # Calculate weighted average
        overall_score = (
            attention_score * 0.35 +
            behavior_score * 0.25 +
            sentiment_score * 0.20 +
            audio_score * 0.20
        )
        
        return min(100, max(0, overall_score))
    
# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 1024  # 16GB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv'}
ALLOWED_HOSTS = ['127.0.0.1', 'localhost']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
        
    file = request.files['video']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
        
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        analyzer = VideoAnalyzer(filepath)
        results = analyzer.analyze()
        
        # Clean up
        os.remove(filepath)
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error in analyze_video: {str(e)}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if threading.current_thread() is not threading.main_thread():
        mpr.set_start_method('spawn')
    app.run(debug=True)
    CORS(app)