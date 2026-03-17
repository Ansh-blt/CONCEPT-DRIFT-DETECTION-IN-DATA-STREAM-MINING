import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from river import tree
from river.drift import ADWIN, PageHinkley, KSWIN
from scipy.stats import multivariate_normal
import io
import base64
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("tab10")

class DriftStreamGenerator:
    def __init__(self, n_features=2, n_classes=3, n_samples=1000, drift_point=400, 
                 drift_width=200, drift_type='sudden', locality='multi_local', 
                 difficulty='swap_cluster', random_state=42):
        self.rng = np.random.RandomState(random_state)
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.drift_point = drift_point
        self.drift_width = drift_width
        self.drift_type = drift_type
        self.locality = locality
        self.difficulty = difficulty
        self._generate_distributions()

    def _generate_distributions(self):
        self.centers_pre = []
        self.covs_pre = []
        
        angles = np.linspace(0, 2*np.pi, self.n_classes, endpoint=False)
        radius = 8.0
        
        for i in range(self.n_classes):
            if self.n_features == 2:
                center = [radius * np.cos(angles[i]), radius * np.sin(angles[i])]
            else:
                center = self.rng.uniform(-10, 10, self.n_features)
            self.centers_pre.append(np.array(center))
            
            cov = np.eye(self.n_features) * 1.5
            if self.n_features >= 2:
                cov[0, 1] = cov[1, 0] = 0.4
            self.covs_pre.append(cov)
        
        self.centers_post = [c.copy() for c in self.centers_pre]
        self.covs_post = [c.copy() for c in self.covs_pre]
        
        self._apply_drift()

    def _apply_drift(self):
        if self.difficulty == 'swap_cluster':
            if self.n_classes >= 2:
                if 'multi' in self.locality:
                    for i in range(0, self.n_classes-1, 2):
                        if i+1 < self.n_classes:
                            temp = self.centers_post[i].copy()
                            self.centers_post[i] = self.centers_post[i+1].copy()
                            self.centers_post[i+1] = temp
                else:
                    if self.n_classes > 1:
                        temp = self.centers_post[0].copy()
                        self.centers_post[0] = self.centers_post[1].copy()
                        self.centers_post[1] = temp
        
        elif self.difficulty == 'emerging_cluster':
            emergence_strength = 12.0
            if 'single' in self.locality:
                direction = self.rng.uniform(-1, 1, self.n_features)
                direction = direction / np.linalg.norm(direction)
                self.centers_post[0] = self.centers_post[0] + direction * emergence_strength
            else:
                for i in range(min(3, self.n_classes)):
                    direction = self.rng.uniform(-1, 1, self.n_features)
                    direction = direction / np.linalg.norm(direction)
                    self.centers_post[i] = self.centers_post[i] + direction * emergence_strength
        
        elif self.difficulty == 'moving_cluster':
            move_distance = 15.0
            for i in range(self.n_classes):
                if 'single' in self.locality and i != 0:
                    continue
                direction = self.rng.uniform(-1, 1, self.n_features)
                direction = direction / np.linalg.norm(direction)
                if 'local' in self.locality:
                    self.centers_post[i] = self.centers_post[i] + direction * (move_distance * 0.6)
                else:
                    self.centers_post[i] = self.centers_post[i] + direction * move_distance
        
        elif self.difficulty == 'splitting_cluster':
            split_distance = 10.0
            for i in range(self.n_classes):
                if 'single' in self.locality and i != 0:
                    continue
                direction1 = self.rng.uniform(-1, 1, self.n_features)
                direction1 = direction1 / np.linalg.norm(direction1)
                self.centers_post[i] = self.centers_pre[i] + direction1 * split_distance
        
        elif self.difficulty == 'merging_cluster':
            if self.n_classes >= 2:
                merge_point = np.mean(self.centers_pre, axis=0)
                if 'single' in self.locality:
                    direction = merge_point - self.centers_pre[0]
                    self.centers_post[0] = self.centers_post[0] + direction * 0.8
                else:
                    for i in range(min(3, self.n_classes)):
                        direction = merge_point - self.centers_pre[i]
                        self.centers_post[i] = self.centers_post[i] + direction * 0.6
        
        elif self.difficulty == 'rotating_cluster':
            if self.n_features == 2:
                rotation_angle = np.pi / 3
                rotation_matrix = np.array([
                    [np.cos(rotation_angle), -np.sin(rotation_angle)],
                    [np.sin(rotation_angle), np.cos(rotation_angle)]
                ])
                for i in range(self.n_classes):
                    if 'single' in self.locality and i != 0:
                        continue
                    self.centers_post[i] = rotation_matrix @ self.centers_pre[i]
        
        elif self.difficulty == 'expanding_cluster':
            for i in range(self.n_classes):
                if 'single' in self.locality and i != 0:
                    continue
                self.covs_post[i] = self.covs_pre[i] * 4.0
        
        elif self.difficulty == 'shrinking_cluster':
            for i in range(self.n_classes):
                if 'single' in self.locality and i != 0:
                    continue
                self.covs_post[i] = self.covs_pre[i] * 0.2

    def _get_interpolated_centers(self, progress):
        if self.drift_type == 'sudden':
            return self.centers_post if progress >= 1.0 else self.centers_pre
        elif self.drift_type == 'gradual':
            smooth_progress = (1 - np.cos(progress * np.pi)) / 2
            centers = []
            for pre, post in zip(self.centers_pre, self.centers_post):
                interpolated = (1 - smooth_progress) * pre + smooth_progress * post
                centers.append(interpolated)
            return centers
        else:
            steps = 10
            step_progress = min(1.0, np.floor(progress * steps) / steps)
            centers = []
            for pre, post in zip(self.centers_pre, self.centers_post):
                interpolated = (1 - step_progress) * pre + step_progress * post
                centers.append(interpolated)
            return centers

    def generate(self):
        data = []
        labels = []
        indices = []
        
        for i in range(self.n_samples):
            if i < self.drift_point:
                progress = 0.0
                centers = self.centers_pre
                covs = self.covs_pre
            elif i > self.drift_point + self.drift_width:
                progress = 1.0
                centers = self.centers_post
                covs = self.covs_post
            else:
                progress = (i - self.drift_point) / self.drift_width
                centers = self._get_interpolated_centers(progress)
                covs = self.covs_post
            
            y = self.rng.randint(0, self.n_classes)
            
            if self.n_features == 1:
                x = self.rng.normal(loc=centers[y][0], scale=np.sqrt(covs[y][0, 0]))
                x = [x]
            else:
                x = self.rng.multivariate_normal(centers[y], covs[y])
            
            data.append(x)
            labels.append(y)
            indices.append(i)
        
        return np.array(data), np.array(labels), np.array(indices)

def run_batch_drift_detection(data, labels, detector_type, batch_size):
    """Run drift detection on batches with a specific detector and reset classifier on drift."""
    if detector_type == 'ADWIN':
        detector = ADWIN()
    elif detector_type == 'PageHinkley':
        detector = PageHinkley()
    else:
        detector = KSWIN()

    classifier = tree.HoeffdingTreeClassifier()

    batch_results = []
    alarms = []
    accuracies = []
    error_rates = []

    correct = 0
    total = 0

    # Rolling stats for normalization
    running_mean = None
    running_var = None
    alpha = 0.01  # EMA smoothing factor

    num_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(data))

        batch_data = data[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]

        batch_correct = 0
        batch_alarms = []

        for i, (x, y) in enumerate(zip(batch_data, batch_labels)):
            x_dict = {f"x{j}": float(x[j]) for j in range(len(x))}

            y_pred = classifier.predict_one(x_dict)
            if y_pred == y:
                correct += 1
                batch_correct += 1
                error = 0
            else:
                error = 1

            total += 1
            error_rates.append(error)
            accuracies.append(correct / total)

            # Compute normalized signal using EMA
            signal = float(np.mean(x))
            if running_mean is None:
                running_mean = signal
                running_var = 1.0
            else:
                running_mean = (1 - alpha) * running_mean + alpha * signal
                running_var = (1 - alpha) * running_var + alpha * (signal - running_mean) ** 2

            std = max(np.sqrt(running_var), 1e-6)
            normalized = abs(signal - running_mean) / std

            detector.update(float(normalized))

            drift_detected = detector.drift_detected

            if drift_detected:
                alarms.append(start_idx + i)
                batch_alarms.append(i)
                classifier = tree.HoeffdingTreeClassifier()

            classifier.learn_one(x_dict, int(y))

        batch_accuracy = batch_correct / len(batch_data) if len(batch_data) > 0 else 0
        batch_results.append({
            'batch_number': batch_idx + 1,
            'start_index': start_idx,
            'end_index': end_idx - 1,
            'batch_size': len(batch_data),
            'batch_accuracy': batch_accuracy,
            'alarms_in_batch': len(batch_alarms),
            'alarm_indices': batch_alarms
        })

    return {
        'detector': detector_type,
        'batch_results': batch_results,
        'total_alarms': len(alarms),
        'alarm_positions': alarms,
        'accuracies': accuracies,
        'error_rates': error_rates,
        'final_accuracy': accuracies[-1] if accuracies else 0,
        'avg_accuracy': float(np.mean(accuracies)) if accuracies else 0
    }

def compute_drift_metrics(alarm_positions, drift_point, drift_width):
    """
    Compute precision, recall, F1-score, and detection delay.
    Alarms within a tolerance window around the drift are counted as TP.
    """
    drift_start = drift_point
    drift_end = drift_point + drift_width

    # Allow detection from the very start up to drift_end + tolerance (late detection)
    tolerance = drift_width
    window_start = 0
    window_end = drift_end + tolerance

    in_drift_alarms = [a for a in alarm_positions if window_start <= a <= window_end]
    out_of_drift_alarms = [a for a in alarm_positions if a > window_end]

    TP = 1 if len(in_drift_alarms) > 0 else 0
    FP = len(out_of_drift_alarms)
    FN = 0 if TP == 1 else 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Detection delay: positive = detected after drift, negative = detected early
    if TP == 1:
        delay = min(in_drift_alarms) - drift_start
    else:
        delay = None

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "delay": delay
    }


def create_detector_plot(result, stream_params, detector_name):
    """Create visualization for a specific detector"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    time_axis = range(len(result['accuracies']))
    drift_start = stream_params['drift_point']
    drift_end = drift_start + stream_params['drift_width']
    
    axes[0].plot(time_axis, result['accuracies'], color='#2E86AB', linewidth=1.5, alpha=0.9)
    axes[0].axvspan(drift_start, drift_end, alpha=0.3, color='#F18F01', label='Drift Region')
    axes[0].axvline(drift_start, color='#F18F01', linestyle='--', linewidth=2, label='Drift Start')
    
    for alarm in result['alarm_positions']:
        axes[0].axvline(alarm, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='lower right')
    axes[0].set_title(f'{detector_name} - Accuracy Over Time', fontsize=14, fontweight='bold')
    
    if len(result['error_rates']) > 1000:
        window = 50
        error_smooth = pd.Series(result['error_rates']).rolling(window=window, center=True).mean()
        axes[1].plot(time_axis, error_smooth, color='#A23B72', linewidth=1.5, alpha=0.8)
    else:
        axes[1].plot(time_axis, result['error_rates'], color='#A23B72', linewidth=1, alpha=0.6)
    
    axes[1].axvspan(drift_start, drift_end, alpha=0.3, color='#F18F01')
    axes[1].axvline(drift_start, color='#F18F01', linestyle='--', linewidth=2)
    
    for alarm in result['alarm_positions']:
        axes[1].axvline(alarm, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    axes[1].set_xlabel('Instance Index', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Error Rate', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title(f'{detector_name} - Error Rate and Drift Alarms', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64

def create_comparison_plot(results, stream_params):
    """Create comparison plot for all three detectors"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    colors = {'ADWIN': '#2E86AB', 'PageHinkley': '#A23B72', 'KSWIN': '#F18F01'}
    
    drift_start = stream_params['drift_point']
    drift_end = drift_start + stream_params['drift_width']
    
    for idx, (detector_name, result) in enumerate(results.items()):
        ax = axes[idx]
        time_axis = range(len(result['accuracies']))
        
        ax.plot(time_axis, result['accuracies'], color=colors[detector_name], 
                linewidth=1.5, alpha=0.9, label=f'{detector_name} Accuracy')
        ax.axvspan(drift_start, drift_end, alpha=0.2, color='gray', label='Drift Region')
        ax.axvline(drift_start, color='black', linestyle='--', linewidth=1.5)
        
        for alarm in result['alarm_positions']:
            ax.axvline(alarm, color='red', linestyle=':', linewidth=1, alpha=0.6)
        
        ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        ax.set_title(f'{detector_name} - Alarms: {result["total_alarms"]}, Avg Accuracy: {result["avg_accuracy"]:.3f}', 
                    fontsize=12, fontweight='bold')
    
    axes[2].set_xlabel('Instance Index', fontsize=12, fontweight='bold')
    plt.suptitle('Drift Detection Comparison - All Algorithms', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/generate', methods=['POST'])
def generate_data():
    params = request.json
    
    n_features = int(params.get('n_features', 2))
    n_classes = int(params.get('n_classes', 3))
    n_samples = int(params.get('n_samples', 1000))
    drift_point = int(params.get('drift_point', n_samples // 3))
    drift_width = int(params.get('drift_width', n_samples // 10))
    drift_type = params.get('drift_type', 'sudden')
    locality = params.get('locality', 'multi_local')
    difficulty = params.get('difficulty', 'swap_cluster')
    
    stream = DriftStreamGenerator(
        n_features=n_features,
        n_classes=n_classes,
        n_samples=n_samples,
        drift_point=drift_point,
        drift_width=drift_width,
        drift_type=drift_type,
        locality=locality,
        difficulty=difficulty
    )
    
    data, labels, indices = stream.generate()
    
    return jsonify({
        'success': True,
        'data': data.tolist(),
        'labels': labels.tolist(),
        'indices': indices.tolist(),
        'params': {
            'n_features': n_features,
            'n_classes': n_classes,
            'n_samples': n_samples,
            'drift_point': drift_point,
            'drift_width': drift_width,
            'drift_type': drift_type,
            'locality': locality,
            'difficulty': difficulty
        }
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_drift():
    try:
        params = request.json
        
        if not params:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        if 'data' not in params or 'labels' not in params or 'stream_params' not in params:
            return jsonify({'success': False, 'error': 'Missing required fields: data, labels, or stream_params'}), 400
        
        data = np.array(params['data'])
        labels = np.array(params['labels'])
        stream_params = params['stream_params']
        
        if len(data) == 0 or len(labels) == 0:
            return jsonify({'success': False, 'error': 'Empty data or labels'}), 400
        
        if len(data) != len(labels):
            return jsonify({'success': False, 'error': 'Data and labels must have the same length'}), 400
        
        batch_size = 100 if len(data) >= 100 else 10
        
        detectors = ['ADWIN', 'PageHinkley', 'KSWIN']
        results = {}
        
        for detector_name in detectors:
            result = run_batch_drift_detection(data, labels, detector_name, batch_size)
            
            metrics = compute_drift_metrics(
        alarm_positions=result["alarm_positions"],
        drift_point=stream_params["drift_point"],
        drift_width=stream_params["drift_width"]
            )
            result["metrics"] = metrics
            results[detector_name] = result
        
        detector_plots = {}
        for detector_name, result in results.items():
            img = create_detector_plot(result, stream_params, detector_name)
            detector_plots[detector_name] = img
        
        comparison_plot = create_comparison_plot(results, stream_params)
        
        simplified_results = {}
        for detector_name, result in results.items():
            simplified_results[detector_name] = {
                'detector': result['detector'],
                'batch_results': result['batch_results'],
                'total_alarms': result['total_alarms'],
                'alarm_positions': result['alarm_positions'],
                'final_accuracy': result['final_accuracy'],
                'avg_accuracy': result['avg_accuracy'],
                'metrics': result['metrics']
            }
        
        return jsonify({
            'success': True,
            'batch_size': batch_size,
            'results': simplified_results,
            'plots': detector_plots,
            'comparison_plot': comparison_plot
        })
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in analyze_drift: {error_details}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
