// Neural Network logic in JS, inspired by your Java code
class Neuron {
    constructor(numInputs) {
        // +1 for bias
        this.weights = Array(numInputs + 1).fill(0).map(() => Math.random() * 2 - 1);
    }
    activate(inputs) {
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
        this.numInputs = numInputs;
        this.numHidden = numHidden;
        this.numOutputs = numOutputs;
        this.inputLayer = Array(numInputs).fill(0);
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
        let hiddenOutputs = this.hiddenLayer.map(neuron => neuron.activate(inputData));
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
    let inputs = [];
    for (let i = 0; i < nn.numInputs; i++) {
        let val = parseFloat(document.getElementById('input-' + i).value);
        inputs.push(isNaN(val) ? 0 : val);
    }
    // Run forward pass, but animate
    let hiddenOutputs = nn.hiddenLayer.map(neuron => neuron.activate(inputs));
    let outputOutputs = nn.outputLayer.map(neuron => neuron.activate(hiddenOutputs));
    let outputs = outputOutputs;
    // Animate
    await animateForward(inputs, hiddenOutputs, outputs);
    outputValues.innerHTML = outputs.map((v, i) => `Output ${i}: <b>${v.toFixed(4)}</b>`).join('<br>');
    outputSection.style.display = '';
};

async function animateForward(inputs, hiddenOutputs, outputs) {
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
    let neuronIdx = 0;
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

// Initial build
buildNetwork();
