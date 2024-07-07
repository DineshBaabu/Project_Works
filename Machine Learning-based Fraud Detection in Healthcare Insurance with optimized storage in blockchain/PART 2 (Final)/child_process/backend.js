const { spawn } = require('child_process');

// Array of numbers to send to Python script
const data = [[1, 1, 1, 0, 0, 0, 0, 1, 0, 36, 790, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 74, 0, 40, 0, 1460, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]];

// Spawn a child process to run the Python script
const pythonProcess = spawn('python', ['model.py']);

// Send data to Python script via stdin
pythonProcess.stdin.write(JSON.stringify(data));
pythonProcess.stdin.end();

// Listen for data from Python script via stdout
pythonProcess.stdout.on('data', (data) => {
    const result = JSON.parse(data.toString());
    console.log('Result from Python script:', result);
});

// Handle errors
pythonProcess.stderr.on('data', (data) => {
    console.error(`Python script error: ${data}`);
});
