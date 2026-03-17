let generatedData = null;
let generatedLabels = null;
let streamParams = null;
let uploadedCSVData = null;

const statusEl = document.getElementById('status');
const batchInfoEl = document.getElementById('batchInfo');
const resultsSectionEl = document.getElementById('resultsSection');
const generateBtn = document.getElementById('generateBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const uploadCsvBtn = document.getElementById('uploadCsvBtn');
const csvFileInput = document.getElementById('csvFileInput');
const processCSVBtn = document.getElementById('processCSVBtn');
const fileNameSpan = document.getElementById('fileName');

function showStatus(message, type = 'info') {
    statusEl.textContent = message;
    statusEl.className = `status-message ${type}`;
    statusEl.style.display = 'block';
}

function hideStatus() {
    statusEl.style.display = 'none';
}

function getFormData() {
    return {
        n_features: parseInt(document.getElementById('n_features').value),
        n_classes: parseInt(document.getElementById('n_classes').value),
        n_samples: parseInt(document.getElementById('n_samples').value),
        drift_point: parseInt(document.getElementById('drift_point').value),
        drift_width: parseInt(document.getElementById('drift_width').value),
        drift_type: document.getElementById('drift_type').value,
        locality: document.getElementById('locality').value,
        difficulty: document.getElementById('difficulty').value
    };
}

generateBtn.addEventListener('click', async () => {
    const params = getFormData();
    
    generateBtn.disabled = true;
    analyzeBtn.disabled = true;
    showStatus('Generating dataset...', 'info');
    
    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        });
        
        const result = await response.json();
        
        if (result.success) {
            generatedData = result.data;
            generatedLabels = result.labels;
            streamParams = result.params;
            
            const batchSize = generatedData.length >= 100 ? 100 : 10;
            const numBatches = Math.ceil(generatedData.length / batchSize);
            
            batchInfoEl.innerHTML = `
                <h4>📦 Dataset Generated Successfully</h4>
                <p><strong>Total Samples:</strong> ${generatedData.length}</p>
                <p><strong>Batch Size:</strong> ${batchSize} (${numBatches} batches)</p>
                <p><strong>Features:</strong> ${streamParams.n_features} | <strong>Classes:</strong> ${streamParams.n_classes}</p>
                <p><strong>Drift Type:</strong> ${streamParams.drift_type} | <strong>Pattern:</strong> ${streamParams.difficulty.replace(/_/g, ' ')}</p>
            `;
            
            showStatus('Dataset generated successfully! Click "Analyze Drift" to process.', 'success');
            analyzeBtn.disabled = false;
            resultsSectionEl.style.display = 'none';
        } else {
            showStatus('Error generating dataset', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showStatus('Error generating dataset: ' + error.message, 'error');
    } finally {
        generateBtn.disabled = false;
    }
});

analyzeBtn.addEventListener('click', async () => {
    if (!generatedData || !generatedLabels) {
        showStatus('Please generate a dataset first', 'error');
        return;
    }
    
    generateBtn.disabled = true;
    analyzeBtn.disabled = true;
    showStatus('Analyzing drift with all algorithms...', 'info');
    resultsSectionEl.style.display = 'none';
    
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                data: generatedData,
                labels: generatedLabels,
                stream_params: streamParams
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayResults(result);
            showStatus('Analysis complete! Results displayed below.', 'success');
        } else {
            showStatus(`Error analyzing drift: ${result.error || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showStatus('Error analyzing drift: ' + error.message, 'error');
    } finally {
        generateBtn.disabled = false;
        analyzeBtn.disabled = false;
    }
});

function displayResults(result) {
    resultsSectionEl.style.display = 'block';
    
    const comparisonPlotEl = document.getElementById('comparisonPlot');
    comparisonPlotEl.innerHTML = `<img src="data:image/png;base64,${result.comparison_plot}" alt="Comparison Plot">`;
    
    displayAlgorithmResults('adwin', 'ADWIN', result.results.ADWIN, result.plots.ADWIN, result.batch_size);
    displayAlgorithmResults('pageHinkley', 'PageHinkley', result.results.PageHinkley, result.plots.PageHinkley, result.batch_size);
    displayAlgorithmResults('kswin', 'KSWIN', result.results.KSWIN, result.plots.KSWIN, result.batch_size);
}

function displayAlgorithmResults(elementPrefix, algoName, algoResult, plotImage, batchSize) {
    const statsEl = document.getElementById(`${elementPrefix}Stats`);
    const plotEl = document.getElementById(`${elementPrefix}Plot`);
    const batchesEl = document.getElementById(`${elementPrefix}Batches`);
    const m = algoResult.metrics;

    statsEl.innerHTML = `
        <div class="stat-row">
            <span class="stat-label">Total Alarms:</span>
            <span class="stat-value">${algoResult.total_alarms}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Final Accuracy:</span>
            <span class="stat-value">${(algoResult.final_accuracy * 100).toFixed(2)}%</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Average Accuracy:</span>
            <span class="stat-value">${(algoResult.avg_accuracy * 100).toFixed(2)}%</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Batch Size Used:</span>
            <span class="stat-value">${batchSize}</span>
        </div>
            <div class="stat-row">
        <span class="stat-label">Precision:</span>
        <span class="stat-value">${m.precision.toFixed(2)}</span>
    </div>
    <div class="stat-row">
        <span class="stat-label">Recall:</span>
        <span class="stat-value">${m.recall.toFixed(2)}</span>
    </div>
    <div class="stat-row">
        <span class="stat-label">F1-score:</span>
        <span class="stat-value">${m.f1_score.toFixed(2)}</span>
    </div>
    <div class="stat-row">
        <span class="stat-label">Detection Delay:</span>
        <span class="stat-value">${m.delay !== null ? m.delay.toFixed(0) : 'N/A'}</span>
    </div>

    `;
    
    plotEl.innerHTML = `<img src="data:image/png;base64,${plotImage}" alt="${algoName} Plot">`;
    
    let batchesHTML = '<h4>Batch-by-Batch Results</h4>';
    
    algoResult.batch_results.forEach(batch => {
        const hasAlarm = batch.alarms_in_batch > 0;
        const alarmClass = hasAlarm ? 'has-alarm' : 'no-alarm';
        
        batchesHTML += `
            <div class="batch-item ${alarmClass}">
                <div class="batch-header">
                    <span>Batch ${batch.batch_number}</span>
                    <span>Accuracy: ${(batch.batch_accuracy * 100).toFixed(2)}%</span>
                </div>
                <div class="batch-details">
                    Indices: ${batch.start_index}-${batch.end_index} | 
                    Size: ${batch.batch_size} | 
                    Alarms: ${batch.alarms_in_batch}
                    ${hasAlarm ? ` (at positions: ${batch.alarm_indices.join(', ')})` : ''}
                </div>
            </div>
        `;
    });
    
    batchesEl.innerHTML = batchesHTML;
}

document.getElementById('n_samples').addEventListener('change', (e) => {
    const nSamples = parseInt(e.target.value);
    document.getElementById('drift_point').value = Math.floor(nSamples / 3);
    document.getElementById('drift_width').value = Math.floor(nSamples / 10);
});

uploadCsvBtn.addEventListener('click', () => {
    csvFileInput.click();
});

csvFileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        fileNameSpan.textContent = file.name;
        processCSVBtn.style.display = 'inline-block';
        
        const reader = new FileReader();
        reader.onload = async (event) => {
            const csvText = event.target.result;
            parseCSV(csvText);
        };
        reader.readAsText(file);
    }
});

function parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    const data = [];
    const labels = [];
    
    let startIndex = 0;
    const firstLine = lines[0].split(',');
    if (isNaN(parseFloat(firstLine[firstLine.length - 1]))) {
        startIndex = 1;
    }
    
    for (let i = startIndex; i < lines.length; i++) {
        const values = lines[i].split(',').map(v => parseFloat(v.trim()));
        if (values.length > 1) {
            const features = values.slice(0, -1);
            const label = Math.round(values[values.length - 1]);
            data.push(features);
            labels.push(label);
        }
    }
    
    uploadedCSVData = { data, labels };
    showStatus(`CSV loaded: ${data.length} samples, ${data[0].length} features`, 'success');
}

processCSVBtn.addEventListener('click', async () => {
    if (!uploadedCSVData) {
        showStatus('Please upload a CSV file first', 'error');
        return;
    }
    
    const driftPoint = parseInt(document.getElementById('csv_drift_point').value);
    const driftWidth = parseInt(document.getElementById('csv_drift_width').value);
    
    generatedData = uploadedCSVData.data;
    generatedLabels = uploadedCSVData.labels;
    streamParams = {
        n_features: generatedData[0].length,
        n_classes: Math.max(...generatedLabels) + 1,
        n_samples: generatedData.length,
        drift_point: driftPoint,
        drift_width: driftWidth,
        drift_type: 'unknown',
        locality: 'custom',
        difficulty: 'custom_csv'
    };
    
    const batchSize = generatedData.length >= 100 ? 100 : 10;
    const numBatches = Math.ceil(generatedData.length / batchSize);
    
    batchInfoEl.innerHTML = `
        <h4> CSV Data Loaded Successfully</h4>
        <p><strong>Total Samples:</strong> ${generatedData.length}</p>
        <p><strong>Batch Size:</strong> ${batchSize} (${numBatches} batches)</p>
        <p><strong>Features:</strong> ${streamParams.n_features} | <strong>Classes:</strong> ${streamParams.n_classes}</p>
        <p><strong>Drift Point:</strong> ${driftPoint} | <strong>Drift Width:</strong> ${driftWidth}</p>
    `;
    
    showStatus('CSV data ready for analysis! Click "Analyze Drift" to process.', 'success');
    analyzeBtn.disabled = false;
    resultsSectionEl.style.display = 'none';
});
