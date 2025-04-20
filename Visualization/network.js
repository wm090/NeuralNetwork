// Neural Network logic in JS, inspired by your Java code
class Neuron {
    constructor(numInputs = null) {
        // If numInputs is null, this is an input neuron
        this.isInputNeuron = (numInputs === null);
        this.value = 0.0;

        if (!this.isInputNeuron) {
            // +1 for bias
            this.weights = Array(numInputs + 1).fill(0).map(() => Math.random() * 2 - 1);
        } else {
            this.weights = null; // Input neurons don't have weights
        }
    }

    setValue(value) {
        this.value = value;
    }

    getValue() {
        return this.value;
    }

    activate(inputs) {
        // For input neurons, just return the stored value
        if (this.isInputNeuron) {
            return this.value;
        }

        // For hidden and output neurons, calculate weighted sum
        let sum = 0;
        for (let i = 0; i < inputs.length; i++) {
            sum += inputs[i] * this.weights[i];
        }
        sum += this.weights[inputs.length]; // bias
        return 1 / (1 + Math.exp(-sum)); // sigmoid
    }
}

class NeuralNetwork {
    constructor(numInputs, numHidden, numOutputs) {
        // Validate network architecture
        if (numInputs <= 0) {
            throw new Error("Network must have at least one input neuron");
        }
        if (numHidden <= 0) {
            throw new Error("Network must have at least one hidden neuron");
        }
        if (numOutputs <= 0) {
            throw new Error("Network must have at least one output neuron");
        }

        this.numInputs = numInputs;
        this.numHidden = numHidden;
        this.numOutputs = numOutputs;

        // Create input neurons (no weights, just pass-through values)
        this.inputLayer = Array(numInputs).fill(0).map(() => new Neuron());

        // Create hidden and output neurons with weights
        this.hiddenLayer = Array(numHidden).fill(0).map(() => new Neuron(numInputs));
        this.outputLayer = Array(numOutputs).fill(0).map(() => new Neuron(numHidden));
    }
    setWeights(hiddenWeights, outputWeights) {
        for (let i = 0; i < this.hiddenLayer.length; i++) {
            if (i < hiddenWeights.length) {
                for (let j = 0; j < this.hiddenLayer[i].weights.length && j < hiddenWeights[i].length; j++) {
                    this.hiddenLayer[i].weights[j] = hiddenWeights[i][j];
                }
            }
        }
        for (let i = 0; i < this.outputLayer.length; i++) {
            if (i < outputWeights.length) {
                for (let j = 0; j < this.outputLayer[i].weights.length && j < outputWeights[i].length; j++) {
                    this.outputLayer[i].weights[j] = outputWeights[i][j];
                }
            }
        }
    }
    forward(inputData) {
        // Validate and clean input data - replace any NaN with 0
        const cleanInputData = inputData.map(val => isNaN(val) ? 0 : val);

        // Step 1: Set values for input neurons
        for (let i = 0; i < this.inputLayer.length; i++) {
            this.inputLayer[i].setValue(cleanInputData[i]);
        }

        // Step 2: Get values from input neurons
        let inputValues = this.inputLayer.map(neuron => neuron.getValue());

        // Step 3: Compute hidden layer activations
        let hiddenOutputs = this.hiddenLayer.map(neuron => neuron.activate(inputValues));

        // Step 4: Compute output layer activations
        let outputOutputs = this.outputLayer.map(neuron => neuron.activate(hiddenOutputs));

        return outputOutputs;
    }
    getWeights() {
        let hiddenWeights = this.hiddenLayer.map(n => [...n.weights]);
        let outputWeights = this.outputLayer.map(n => [...n.weights]);
        return [hiddenWeights, outputWeights];
    }
}

// --- UI Logic ---
let nn = null;
let inputFields = document.getElementById('input-fields');
let weightsFields = document.getElementById('weights-fields');
let outputValues = document.getElementById('output-values');
let inputSection = document.getElementById('input-section');
let weightsSection = document.getElementById('weights-section');
let outputSection = document.getElementById('output-section');
let visualization = document.getElementById('visualization');

function buildNetwork() {
    let numInputs = parseInt(document.getElementById('input-neurons').value);
    let numHidden = parseInt(document.getElementById('hidden-neurons').value);
    let numOutputs = parseInt(document.getElementById('output-neurons').value);

    // Validate network architecture (matching Java validation)
    try {
        if (numInputs <= 0) {
            throw new Error("Network must have at least one input neuron");
        }
        if (numHidden <= 0) {
            throw new Error("Network must have at least one hidden neuron");
        }
        if (numOutputs <= 0) {
            throw new Error("Network must have at least one output neuron");
        }

        // If validation passes, create the network
        nn = new NeuralNetwork(numInputs, numHidden, numOutputs);
        renderInputs(numInputs);
        renderWeights();
        renderVisualization();
        inputSection.style.display = '';
        weightsSection.style.display = '';
        outputSection.style.display = 'none';
        // Show manual weights section and clear textareas
        document.getElementById('manual-weights-section').style.display = '';
        document.getElementById('manual-hidden-weights').value = '';
        document.getElementById('manual-output-weights').value = '';
    } catch (error) {
        // Show error message
        alert("Error: " + error.message);
    }
}

document.getElementById('build-network').onclick = buildNetwork;

function renderInputs(numInputs) {
    inputFields.innerHTML = '';
    for (let i = 0; i < numInputs; i++) {
        let inp = document.createElement('input');
        inp.type = 'number';
        inp.step = 'any';
        inp.value = 0;
        inp.id = 'input-' + i;
        inputFields.appendChild(inp);
    }
}

function renderWeights() {
    weightsFields.innerHTML = '';
    // Hidden weights
    let [hiddenWeights, outputWeights] = nn.getWeights();
    let hiddenDiv = document.createElement('div');
    hiddenDiv.innerHTML = '<b>Hidden Weights</b>';
    for (let i = 0; i < hiddenWeights.length; i++) {
        let row = document.createElement('div');
        row.innerText = `Neuron ${i}: `;
        for (let j = 0; j < hiddenWeights[i].length; j++) {
            let w = document.createElement('input');
            w.type = 'number';
            w.step = 'any';
            w.value = hiddenWeights[i][j];
            w.dataset.layer = 'hidden';
            w.dataset.i = i;
            w.dataset.j = j;
            w.onchange = updateWeight;
            row.appendChild(w);
        }
        hiddenDiv.appendChild(row);
    }
    weightsFields.appendChild(hiddenDiv);
    // Output weights
    let outputDiv = document.createElement('div');
    outputDiv.innerHTML = '<b>Output Weights</b>';
    for (let i = 0; i < outputWeights.length; i++) {
        let row = document.createElement('div');
        row.innerText = `Neuron ${i}: `;
        for (let j = 0; j < outputWeights[i].length; j++) {
            let w = document.createElement('input');
            w.type = 'number';
            w.step = 'any';
            w.value = outputWeights[i][j];
            w.dataset.layer = 'output';
            w.dataset.i = i;
            w.dataset.j = j;
            w.onchange = updateWeight;
            row.appendChild(w);
        }
        outputDiv.appendChild(row);
    }
    weightsFields.appendChild(outputDiv);
}

function updateWeight(e) {
    let layer = e.target.dataset.layer;
    let i = parseInt(e.target.dataset.i);
    let j = parseInt(e.target.dataset.j);
    let val = parseFloat(e.target.value);
    let [hiddenWeights, outputWeights] = nn.getWeights();
    if (layer === 'hidden') {
        hiddenWeights[i][j] = val;
    } else {
        outputWeights[i][j] = val;
    }
    nn.setWeights(hiddenWeights, outputWeights);
    renderVisualization();
}

document.getElementById('randomize-weights').onclick = function() {
    buildNetwork(); // Rebuilds with new random weights
}

document.getElementById('run-forward').onclick = async function() {
    // Get input values from UI
    let inputs = [];
    for (let i = 0; i < nn.numInputs; i++) {
        let val = parseFloat(document.getElementById('input-' + i).value);
        inputs.push(isNaN(val) ? 0 : val);
    }

    // Set values for input neurons
    for (let i = 0; i < nn.inputLayer.length; i++) {
        nn.inputLayer[i].setValue(inputs[i]);
    }

    // Get values from input neurons
    let inputValues = nn.inputLayer.map(neuron => neuron.getValue());

    // Compute hidden layer activations
    let hiddenOutputs = nn.hiddenLayer.map(neuron => neuron.activate(inputValues));

    // Compute output layer activations
    let outputs = nn.outputLayer.map(neuron => neuron.activate(hiddenOutputs));

    // Animate
    await animateForward(inputs, hiddenOutputs, outputs);

    // Display results
    outputValues.innerHTML = outputs.map((v, i) => `Output ${i}: <b>${v.toFixed(4)}</b>`).join('<br>');
    outputSection.style.display = '';
};

// Animation function for the forward pass
// Note: hiddenOutputs parameter is kept for consistency with the forward pass calculation
// but is not used in the animation itself
async function animateForward(inputs, _hiddenOutputs, outputs) {
    // Animate input neurons
    renderVisualization(inputs, []);
    highlightNeurons(0);
    await sleep(500);
    // Animate input->hidden connections
    highlightConnections(0, 1);
    await sleep(400);
    // Animate hidden neurons
    renderVisualization(inputs, []);
    highlightNeurons(1);
    await sleep(500);
    // Animate hidden->output connections
    highlightConnections(1, 2);
    await sleep(400);
    // Animate output neurons
    renderVisualization(inputs, outputs);
    highlightNeurons(2);
    await sleep(500);
    // Remove highlights
    renderVisualization(inputs, outputs);
}

function sleep(ms) {
    return new Promise(res => setTimeout(res, ms));
}

function highlightNeurons(layerIdx) {
    let svg = visualization.querySelector('svg');
    if (!svg) return;
    let circles = svg.querySelectorAll('circle.neuron');
    let layers = [nn.numInputs, nn.numHidden, nn.numOutputs];
    let idx = 0;
    for (let l = 0; l < layers.length; l++) {
        for (let n = 0; n < layers[l]; n++) {
            if (l === layerIdx) {
                circles[idx].classList.add('highlight-neuron');
            } else {
                circles[idx].classList.remove('highlight-neuron');
            }
            idx++;
        }
    }
}

// Clear all highlights
function clearAllHighlights() {
    let svg = visualization.querySelector('svg');
    if (!svg) return;

    // Clear neuron highlights
    let circles = svg.querySelectorAll('circle.neuron');
    circles.forEach(circle => {
        circle.classList.remove('highlight-neuron');
    });

    // Clear connection highlights
    let lines = svg.querySelectorAll('line.connection');
    lines.forEach(line => {
        line.classList.remove('highlight-connection');
    });
}

function highlightConnections(fromLayer, toLayer) {
    let svg = visualization.querySelector('svg');
    if (!svg) return;
    let lines = svg.querySelectorAll('line.connection');
    // Input->Hidden: lines 0..(numInputs*numHidden-1), Hidden->Output: rest
    let numInputs = nn.numInputs, numHidden = nn.numHidden, numOutputs = nn.numOutputs;
    let start = 0, end = 0;
    if (fromLayer === 0 && toLayer === 1) {
        start = 0;
        end = numInputs * numHidden;
    } else if (fromLayer === 1 && toLayer === 2) {
        start = numInputs * numHidden;
        end = start + numHidden * numOutputs;
    }
    for (let i = 0; i < lines.length; i++) {
        if (i >= start && i < end) {
            lines[i].classList.add('highlight-connection');
        } else {
            lines[i].classList.remove('highlight-connection');
        }
    }
}


function renderVisualization(inputs = [], outputs = []) {
    // Draw a simple SVG network
    let w = 400, h = 250;
    let svg = `<svg width="${w}" height="${h}">
`;
    let layers = [nn.numInputs, nn.numHidden, nn.numOutputs];
    let xStep = w / (layers.length + 1);
    let ySteps = layers.map(n => h / (n + 1));
    let positions = [];
    for (let l = 0; l < layers.length; l++) {
        let arr = [];
        for (let n = 0; n < layers[l]; n++) {
            arr.push({
                x: xStep * (l + 1),
                y: ySteps[l] * (n + 1)
            });
        }
        positions.push(arr);
    }
    // Draw connections
    // Input->Hidden
    for (let i = 0; i < positions[0].length; i++) {
        for (let j = 0; j < positions[1].length; j++) {
            svg += `<line class='connection' x1='${positions[0][i].x}' y1='${positions[0][i].y}' x2='${positions[1][j].x}' y2='${positions[1][j].y}' />`;
        }
    }
    // Hidden->Output
    for (let i = 0; i < positions[1].length; i++) {
        for (let j = 0; j < positions[2].length; j++) {
            svg += `<line class='connection' x1='${positions[1][i].x}' y1='${positions[1][i].y}' x2='${positions[2][j].x}' y2='${positions[2][j].y}' />`;
        }
    }
    // Draw neurons
    for (let l = 0; l < layers.length; l++) {
        for (let n = 0; n < layers[l]; n++) {
            let val = '';
            if (l === 0 && inputs.length) val = inputs[n]?.toFixed(2);
            if (l === 2 && outputs.length) val = outputs[n]?.toFixed(2);
            svg += `<circle class='neuron' cx='${positions[l][n].x}' cy='${positions[l][n].y}' r='16'/><text x='${positions[l][n].x}' y='${positions[l][n].y+5}' text-anchor='middle' font-size='12'>${val}</text>`;
        }
    }
    svg += '</svg>';
    visualization.innerHTML = `<div class='network'>${svg}</div>`;
}

// Manual weights logic

document.getElementById('apply-manual-weights').onclick = function() {
    try {
        let hiddenStr = document.getElementById('manual-hidden-weights').value.trim();
        let outputStr = document.getElementById('manual-output-weights').value.trim();
        let hiddenWeights = hiddenStr ? JSON.parse(hiddenStr) : [];
        let outputWeights = outputStr ? JSON.parse(outputStr) : [];
        if (!Array.isArray(hiddenWeights) || !Array.isArray(outputWeights)) throw new Error();
        nn.setWeights(hiddenWeights, outputWeights);
        renderWeights();
        renderVisualization();
        alert('Weights applied successfully.');
    } catch (e) {
        alert('Failed to parse weights. Please use valid JSON array format.');
    }
};

// File processing variables
let trainingData = [];
let weightsData = null;
let trainingNetwork = null;

// Function to read and parse CSV files
async function readCSVFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const content = e.target.result;
            resolve(content);
        };
        reader.onerror = () => {
            reject(new Error('Error reading file'));
        };
        reader.readAsText(file);
    });
}

// Parse training data CSV
function parseTrainingData(csvContent) {
    const lines = csvContent.split('\n').filter(line => line.trim() !== '');
    const parsedData = [];

    for (const line of lines) {
        // Split by tab or multiple spaces
        const parts = line.split(/\s+/);

        if (parts.length >= 2) {
            // Parse inputs (RGB values)
            const inputParts = parts[0].split(';');
            const inputs = inputParts.map(val => {
                const parsed = parseFloat(val.trim());
                // Replace NaN with 0 to avoid issues
                return isNaN(parsed) ? 0 : parsed;
            });

            // Parse expected outputs (traffic light class)
            const outputParts = parts[1].split(';');
            const expectedOutputs = outputParts.map(val => {
                const parsed = parseFloat(val.trim());
                // Replace NaN with 0 to avoid issues
                return isNaN(parsed) ? 0 : parsed;
            });

            // Validate that we have valid data
            if (inputs.length > 0 && expectedOutputs.length > 0) {
                // Add to parsed data
                parsedData.push({
                    inputs,
                    expectedOutputs
                });
            }
        }
    }

    return parsedData;
}

// Parse weights CSV
function parseWeightsCSV(csvContent) {
    const lines = csvContent.split('\n').filter(line => line.trim() !== '');

    // First line contains network dimensions
    const dimensions = lines[0].split(';');
    if (dimensions.length < 3 || dimensions[0] !== 'layers') {
        throw new Error('Invalid weights file format');
    }

    const numInputs = parseInt(dimensions[1]);
    const numHidden = parseInt(dimensions[2]);
    const numOutputs = parseInt(dimensions[3]);

    // Parse hidden layer weights
    const hiddenWeights = [];
    for (let i = 1; i <= numHidden; i++) {
        if (i < lines.length) {
            const weightValues = lines[i].split(';').map(val => {
                const trimmed = val.trim();
                return trimmed ? parseFloat(trimmed) : 0;
            });
            hiddenWeights.push(weightValues.slice(0, numInputs + 1)); // +1 for bias
        }
    }

    // Find separator line
    let separatorIndex = -1;
    for (let i = 0; i < lines.length; i++) {
        if (lines[i].trim() === ';;;') {
            separatorIndex = i;
            break;
        }
    }

    // Parse output layer weights
    const outputWeights = [];
    if (separatorIndex !== -1) {
        for (let i = separatorIndex + 1; i < separatorIndex + 1 + numOutputs && i < lines.length; i++) {
            const weightValues = lines[i].split(';').map(val => {
                const trimmed = val.trim();
                return trimmed ? parseFloat(trimmed) : 0;
            });
            outputWeights.push(weightValues.slice(0, numHidden + 1)); // +1 for bias
        }
    }

    return {
        dimensions: { numInputs, numHidden, numOutputs },
        hiddenWeights,
        outputWeights
    };
}

// Process uploaded files
async function processFiles() {
    const trainingFile = document.getElementById('training-data-file').files[0];
    const weightsFile = document.getElementById('weights-file').files[0];
    const resultsContent = document.getElementById('results-content');
    const processingResults = document.getElementById('processing-results');
    const backpropAnimation = document.getElementById('backprop-animation');

    // Clear previous results
    resultsContent.innerHTML = '';
    backpropAnimation.innerHTML = '';
    backpropAnimation.style.display = 'none';
    document.getElementById('run-animation').style.display = 'none';
    document.getElementById('run-training').style.display = 'none';

    if (!trainingFile || !weightsFile) {
        resultsContent.innerHTML += '<div class="error">Please select both training data and weights files.</div>';
        processingResults.style.display = '';
        return;
    }

    try {
        // Read and parse training data
        const trainingCSV = await readCSVFile(trainingFile);
        trainingData = parseTrainingData(trainingCSV);
        resultsContent.innerHTML += `<div class="success">✓ Loaded ${trainingData.length} training examples</div>`;

        // Read and parse weights
        const weightsCSV = await readCSVFile(weightsFile);
        weightsData = parseWeightsCSV(weightsCSV);
        resultsContent.innerHTML += `<div class="success">✓ Loaded weights for network with ${weightsData.dimensions.numInputs} inputs, ${weightsData.dimensions.numHidden} hidden neurons, and ${weightsData.dimensions.numOutputs} outputs</div>`;

        // Create neural network
        if (trainingData.length > 0) {
            const firstExample = trainingData[0];
            const numInputs = firstExample.inputs.length;
            const numOutputs = firstExample.expectedOutputs.length;
            const numHidden = weightsData.dimensions.numHidden;

            resultsContent.innerHTML += `<div class="info">Creating neural network with:\n - ${numInputs} input neurons\n - ${numHidden} hidden neurons\n - ${numOutputs} output neurons</div>`;

            // Create the network
            trainingNetwork = new NeuralNetwork(numInputs, numHidden, numOutputs);

            // Set weights
            trainingNetwork.setWeights(weightsData.hiddenWeights, weightsData.outputWeights);
            resultsContent.innerHTML += `<div class="success">✓ Weights loaded successfully</div>`;

            // Test network before training
            resultsContent.innerHTML += `<div class="info">Testing network BEFORE training:</div>`;
            const beforeResults = testNetwork(trainingNetwork, trainingData);
            resultsContent.innerHTML += `<div>${beforeResults.log}</div>`;
            resultsContent.innerHTML += `<div class="info">Accuracy: ${beforeResults.accuracy.toFixed(2)}% (${beforeResults.correct}/${trainingData.length})</div>`;

            // Show the Run Animation and Run Training buttons
            document.getElementById('run-animation').style.display = '';
            document.getElementById('run-training').style.display = '';

            // Update visualization with the network
            updateVisualizationWithTrainedNetwork();
        }

        processingResults.style.display = '';
    } catch (error) {
        resultsContent.innerHTML += `<div class="error">Error: ${error.message}</div>`;
        processingResults.style.display = '';
    }
}

// Test the network on training data
function testNetwork(network, testData) {
    let correct = 0;
    let log = '';

    for (const data of testData) {
        // Get inputs and expected outputs
        const inputs = data.inputs.map(val => isNaN(val) ? 0 : val); // Handle NaN values
        const expectedOutputs = data.expectedOutputs.map(val => isNaN(val) ? 0 : val); // Handle NaN values

        // Run forward pass
        const actualOutputs = network.forward(inputs);

        // Find which class has the highest value
        const expectedClass = findMaxIndex(expectedOutputs);
        const actualClass = findMaxIndex(actualOutputs);

        // Format the inputs for display, replacing NaN with 0
        const formattedInputs = inputs.map(v => isNaN(v) ? "0.00" : v.toFixed(2));

        // Log the result
        let resultLine = `Input: [${formattedInputs.join(', ')}] | Expected: ${expectedClass} | Actual: ${actualClass}`;

        // Check if prediction was correct
        if (expectedClass === actualClass) {
            resultLine += ' ✓';
            correct++;
        } else {
            resultLine += ' ✗';
        }

        log += resultLine + '\n';
    }

    // Calculate accuracy
    const accuracy = (correct / testData.length) * 100;

    return { log, correct, accuracy };
}

// Find the index of the maximum value in an array
function findMaxIndex(array) {
    let maxIndex = 0;
    let maxValue = array[0];

    for (let i = 1; i < array.length; i++) {
        if (array[i] > maxValue) {
            maxValue = array[i];
            maxIndex = i;
        }
    }

    return maxIndex;
}

// Train the network
function trainNetwork() {
    if (!trainingNetwork || trainingData.length === 0) {
        alert('Please process training data and weights files first.');
        return;
    }

    const resultsContent = document.getElementById('results-content');
    resultsContent.innerHTML += `<div class="info">Training the network...</div>`;

    // Train the network
    const learningRate = 0.1;
    const epochs = 1000;

    // Implement proper backpropagation
    let totalError = 0;

    // Show progress for some epochs
    const progressEpochs = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999];

    for (let epoch = 0; epoch < epochs; epoch++) {
        totalError = 0;

        // Process each training example
        for (const data of trainingData) {
            // Forward pass
            const inputs = data.inputs;
            const expectedOutputs = data.expectedOutputs;

            // Set values for input neurons
            for (let i = 0; i < trainingNetwork.inputLayer.length; i++) {
                trainingNetwork.inputLayer[i].setValue(inputs[i]);
            }

            // Get values from input neurons
            const inputValues = trainingNetwork.inputLayer.map(neuron => neuron.getValue());

            // Compute hidden layer activations
            const hiddenOutputs = [];
            for (let i = 0; i < trainingNetwork.hiddenLayer.length; i++) {
                hiddenOutputs[i] = trainingNetwork.hiddenLayer[i].activate(inputValues);
            }

            // Compute output layer activations
            const actualOutputs = [];
            for (let i = 0; i < trainingNetwork.outputLayer.length; i++) {
                actualOutputs[i] = trainingNetwork.outputLayer[i].activate(hiddenOutputs);
            }

            // Calculate error
            let error = 0;
            for (let i = 0; i < actualOutputs.length; i++) {
                error += Math.pow(expectedOutputs[i] - actualOutputs[i], 2);
            }
            totalError += error;

            // Backpropagation - implement actual weight updates
            // 1. Calculate output layer deltas
            const outputDeltas = [];
            for (let i = 0; i < trainingNetwork.outputLayer.length; i++) {
                const output = actualOutputs[i];
                const target = expectedOutputs[i];
                // Error derivative * sigmoid derivative
                outputDeltas[i] = (target - output) * output * (1 - output);
            }

            // 2. Calculate hidden layer deltas
            const hiddenDeltas = [];
            for (let i = 0; i < trainingNetwork.hiddenLayer.length; i++) {
                let error = 0;
                for (let j = 0; j < trainingNetwork.outputLayer.length; j++) {
                    error += outputDeltas[j] * trainingNetwork.outputLayer[j].weights[i];
                }
                hiddenDeltas[i] = error * hiddenOutputs[i] * (1 - hiddenOutputs[i]);
            }

            // 3. Update output layer weights
            for (let i = 0; i < trainingNetwork.outputLayer.length; i++) {
                for (let j = 0; j < trainingNetwork.hiddenLayer.length; j++) {
                    trainingNetwork.outputLayer[i].weights[j] += learningRate * outputDeltas[i] * hiddenOutputs[j];
                }
                // Update bias weight
                trainingNetwork.outputLayer[i].weights[trainingNetwork.hiddenLayer.length] += learningRate * outputDeltas[i];
            }

            // 4. Update hidden layer weights
            for (let i = 0; i < trainingNetwork.hiddenLayer.length; i++) {
                for (let j = 0; j < trainingNetwork.inputLayer.length; j++) {
                    trainingNetwork.hiddenLayer[i].weights[j] += learningRate * hiddenDeltas[i] * inputs[j];
                }
                // Update bias weight
                trainingNetwork.hiddenLayer[i].weights[trainingNetwork.inputLayer.length] += learningRate * hiddenDeltas[i];
            }
        }

        // Log progress for selected epochs
        if (progressEpochs.includes(epoch)) {
            resultsContent.innerHTML += `<div class="info">Epoch ${epoch}, Error: ${totalError.toFixed(6)}</div>`;
        }
    }

    resultsContent.innerHTML += `<div class="success">✓ Training completed</div>`;

    // Test network after training
    resultsContent.innerHTML += `<div class="info">Testing network AFTER training:</div>`;
    const afterResults = testNetwork(trainingNetwork, trainingData);
    resultsContent.innerHTML += `<div>${afterResults.log}</div>`;
    resultsContent.innerHTML += `<div class="info">Accuracy: ${afterResults.accuracy.toFixed(2)}% (${afterResults.correct}/${trainingData.length})</div>`;

    // Update visualization with the trained network
    updateVisualizationWithTrainedNetwork();
}

// Update visualization with the trained network
function updateVisualizationWithTrainedNetwork() {
    if (!trainingNetwork) return;

    // Update the network size inputs
    document.getElementById('input-neurons').value = trainingNetwork.numInputs;
    document.getElementById('hidden-neurons').value = trainingNetwork.numHidden;
    document.getElementById('output-neurons').value = trainingNetwork.numOutputs;

    // Build the network in the visualization
    buildNetwork();

    // Set the weights from the trained network
    const [hiddenWeights, outputWeights] = trainingNetwork.getWeights();
    nn.setWeights(hiddenWeights, outputWeights);

    // Update the visualization
    renderWeights();
    renderVisualization();

    // Show the input section
    inputSection.style.display = '';
    weightsSection.style.display = '';
}

// Animation state variables
let animationPaused = false;
let currentStep = 0;
let totalSteps = 7;
let animationData = null;
let animationInterval = null;

// Animation for a single forward and backpropagation pass
async function animateForwardAndBackprop() {
    if (!trainingNetwork || trainingData.length === 0) {
        alert('Please process training data and weights files first.');
        return;
    }

    // Reset animation state
    animationPaused = false;
    currentStep = 0;

    // Show animation containers
    const backpropAnimation = document.getElementById('backprop-animation');
    const animationPlayer = document.getElementById('animation-player');
    backpropAnimation.innerHTML = '<h3>Neural Network Animation</h3>';
    backpropAnimation.style.display = '';
    animationPlayer.style.display = '';

    // Make sure the visualization is visible
    document.getElementById('visualization').style.display = '';

    // Update player controls
    document.getElementById('play-pause').innerHTML = '⏸️ Pause';
    document.getElementById('step-back').disabled = true;
    document.getElementById('step-forward').disabled = false;
    document.getElementById('restart').disabled = false;

    // Select the first training example for animation
    const example = trainingData[0];
    const inputs = example.inputs.map(val => isNaN(val) ? 0 : val);
    const expectedOutputs = example.expectedOutputs.map(val => isNaN(val) ? 0 : val);

    // Store animation data
    animationData = { inputs, expectedOutputs };

    // Create animation steps
    backpropAnimation.innerHTML += `
        <div class="animation-step" id="step-1">Step 1: Input values [${inputs.map(v => v.toFixed(2)).join(', ')}]</div>
        <div class="animation-step" id="step-2">Step 2: Forward pass through hidden layer</div>
        <div class="animation-step" id="step-3">Step 3: Forward pass through output layer</div>
        <div class="animation-step" id="step-4">Step 4: Calculate output error</div>
        <div class="animation-step" id="step-5">Step 5: Backpropagate error to hidden layer</div>
        <div class="animation-step" id="step-6">Step 6: Update weights</div>
        <div class="animation-step" id="step-7">Step 7: Final prediction</div>
    `;

    // Clear any existing highlights and render the network
    renderVisualization(inputs, []);
    clearAllHighlights();

    // Scroll to the visualization
    document.getElementById('visualization').scrollIntoView({ behavior: 'smooth' });

    // Start the animation
    await runAnimationStep(1);
}

// Run a specific animation step
async function runAnimationStep(step) {
    if (!animationData) return;

    // Update current step
    currentStep = step;

    // Update progress indicator
    updateProgressIndicator();

    // Clear all active steps
    for (let i = 1; i <= totalSteps; i++) {
        const stepElement = document.getElementById(`step-${i}`);
        if (stepElement) stepElement.classList.remove('active');
    }

    // Get animation data
    const { inputs, expectedOutputs } = animationData;

    // Activate current step
    const currentStepElement = document.getElementById(`step-${step}`);
    if (currentStepElement) currentStepElement.classList.add('active');

    // Clear any existing highlights
    renderVisualization(inputs, animationData.actualOutputs || []);
    clearAllHighlights();

    // Run the appropriate step
    switch (step) {
        case 1: // Input values
            // Highlight input neurons
            highlightNeurons(0);

            // Set values for input neurons
            for (let i = 0; i < trainingNetwork.inputLayer.length; i++) {
                trainingNetwork.inputLayer[i].setValue(inputs[i]);
            }
            break;

        case 2: // Forward pass through hidden layer
            // Get values from input neurons
            const inputValues = trainingNetwork.inputLayer.map(neuron => neuron.getValue());

            // Compute hidden layer activations
            const hiddenOutputs = [];
            for (let i = 0; i < trainingNetwork.hiddenLayer.length; i++) {
                hiddenOutputs[i] = trainingNetwork.hiddenLayer[i].activate(inputValues);
            }

            // Store hidden outputs in animation data
            animationData.hiddenOutputs = hiddenOutputs;

            // Highlight connections from input to hidden
            highlightConnections(0, 1);
            await sleep(500);

            // Highlight hidden neurons
            highlightNeurons(1);
            break;

        case 3: // Forward pass through output layer
            // Make sure we have hidden outputs
            if (!animationData.hiddenOutputs) {
                // Calculate them if they don't exist
                const inputValues = trainingNetwork.inputLayer.map(neuron => neuron.getValue());
                const hiddenOutputs = [];
                for (let i = 0; i < trainingNetwork.hiddenLayer.length; i++) {
                    hiddenOutputs[i] = trainingNetwork.hiddenLayer[i].activate(inputValues);
                }
                animationData.hiddenOutputs = hiddenOutputs;
            }

            // Compute output layer activations
            const actualOutputs = [];
            for (let i = 0; i < trainingNetwork.outputLayer.length; i++) {
                actualOutputs[i] = trainingNetwork.outputLayer[i].activate(animationData.hiddenOutputs);
            }

            // Store actual outputs in animation data
            animationData.actualOutputs = actualOutputs;

            // Highlight connections from hidden to output
            highlightConnections(1, 2);
            await sleep(500);

            // Highlight output neurons and show output values
            renderVisualization(inputs, actualOutputs);
            highlightNeurons(2);
            break;

        case 4: // Calculate output error
            // Make sure we have actual outputs
            if (!animationData.actualOutputs) {
                // We need to calculate them
                await runAnimationStep(3);
            }

            // Calculate error
            let error = 0;
            for (let i = 0; i < animationData.actualOutputs.length; i++) {
                error += Math.pow(expectedOutputs[i] - animationData.actualOutputs[i], 2);
            }

            // Store error in animation data
            animationData.error = error;

            // Display error
            if (!currentStepElement.innerHTML.includes('Error:')) {
                currentStepElement.innerHTML += `<br>Error: ${error.toFixed(4)}`;
            }

            // Highlight output neurons to show where error is calculated
            highlightNeurons(2);
            break;

        case 5: // Backpropagate error
            // Make sure we have actual outputs and hidden outputs
            if (!animationData.actualOutputs || !animationData.hiddenOutputs) {
                // We need to calculate them
                await runAnimationStep(3);
            }

            // Calculate output layer deltas
            const outputDeltas = [];
            for (let i = 0; i < trainingNetwork.outputLayer.length; i++) {
                const output = animationData.actualOutputs[i];
                const target = expectedOutputs[i];
                outputDeltas[i] = (target - output) * output * (1 - output);
            }

            // Calculate hidden layer deltas
            const hiddenDeltas = [];
            for (let i = 0; i < trainingNetwork.hiddenLayer.length; i++) {
                let error = 0;
                for (let j = 0; j < trainingNetwork.outputLayer.length; j++) {
                    error += outputDeltas[j] * trainingNetwork.outputLayer[j].weights[i];
                }
                hiddenDeltas[i] = error * animationData.hiddenOutputs[i] * (1 - animationData.hiddenOutputs[i]);
            }

            // Store deltas in animation data
            animationData.outputDeltas = outputDeltas;
            animationData.hiddenDeltas = hiddenDeltas;

            // First highlight output neurons where backpropagation starts
            highlightNeurons(2);
            await sleep(500);

            // Then highlight connections from output to hidden (backpropagation)
            highlightConnections(1, 2);
            await sleep(500);

            // Finally highlight hidden neurons where deltas are calculated
            highlightNeurons(1);
            break;

        case 6: // Update weights
            // Make sure we have deltas
            if (!animationData.outputDeltas || !animationData.hiddenDeltas) {
                // We need to calculate them
                await runAnimationStep(5);
            }

            // Display weight update information
            if (!currentStepElement.innerHTML.includes('Weights updated')) {
                currentStepElement.innerHTML += `<br>Weights updated using learning rate: 0.1`;
            }

            // Highlight all connections to show weight updates
            highlightConnections(0, 1);
            await sleep(500);
            highlightConnections(1, 2);
            break;

        case 7: // Final prediction
            // Make sure we have actual outputs
            if (!animationData.actualOutputs) {
                // We need to calculate them
                await runAnimationStep(3);
            }

            // Find which class has the highest value
            const expectedClass = findMaxIndex(expectedOutputs);
            const actualClass = findMaxIndex(animationData.actualOutputs);

            // Display prediction
            if (!currentStepElement.innerHTML.includes('Expected class')) {
                currentStepElement.innerHTML += `<br>Expected class: ${expectedClass}, Predicted class: ${actualClass}`;
                if (expectedClass === actualClass) {
                    currentStepElement.innerHTML += ` ✓`;
                } else {
                    currentStepElement.innerHTML += ` ✗`;
                }
            }

            // Show the final output
            renderVisualization(inputs, animationData.actualOutputs);
            highlightNeurons(2);

            // Disable next button if this is the last step
            document.getElementById('step-forward').disabled = true;
            break;
    }

    // If not paused and not the last step, continue to next step after delay
    if (!animationPaused && step < totalSteps) {
        await sleep(1500);
        await runAnimationStep(step + 1);
    }

    // If this is the last step, show completion message
    if (step === totalSteps) {
        document.getElementById('play-pause').innerHTML = '▶️ Play';
        animationPaused = true;
    }
}

// Update the progress indicator
function updateProgressIndicator() {
    // Update progress bar
    const progressPercent = (currentStep / totalSteps) * 100;
    document.getElementById('progress-indicator').style.width = `${progressPercent}%`;

    // Update step indicator
    document.getElementById('step-indicator').textContent = `Step ${currentStep}/${totalSteps}`;

    // Update button states
    document.getElementById('step-back').disabled = currentStep <= 1;
    document.getElementById('step-forward').disabled = currentStep >= totalSteps;
}

// Play/pause the animation
function togglePlayPause() {
    animationPaused = !animationPaused;

    if (animationPaused) {
        document.getElementById('play-pause').innerHTML = '▶️ Play';
    } else {
        document.getElementById('play-pause').innerHTML = '⏸️ Pause';
        // Continue from current step
        if (currentStep < totalSteps) {
            runAnimationStep(currentStep + 1);
        } else {
            // If at the end, restart
            runAnimationStep(1);
        }
    }
}

// Step backward in the animation
async function stepBack() {
    if (currentStep > 1) {
        animationPaused = true;
        document.getElementById('play-pause').innerHTML = '▶️ Play';
        await runAnimationStep(currentStep - 1);
    }
}

// Step forward in the animation
async function stepForward() {
    if (currentStep < totalSteps) {
        animationPaused = true;
        document.getElementById('play-pause').innerHTML = '▶️ Play';
        await runAnimationStep(currentStep + 1);
    }
}

// Restart the animation
async function restartAnimation() {
    animationPaused = false;
    document.getElementById('play-pause').innerHTML = '⏸️ Pause';
    await runAnimationStep(1);
}

// Event listeners for file processing and animation
document.getElementById('process-files').addEventListener('click', processFiles);
document.getElementById('run-animation').addEventListener('click', animateForwardAndBackprop);
document.getElementById('run-training').addEventListener('click', trainNetwork);

// Event listeners for player controls
document.getElementById('play-pause').addEventListener('click', togglePlayPause);
document.getElementById('step-back').addEventListener('click', stepBack);
document.getElementById('step-forward').addEventListener('click', stepForward);
document.getElementById('restart').addEventListener('click', restartAnimation);

// Initial build
buildNetwork();
