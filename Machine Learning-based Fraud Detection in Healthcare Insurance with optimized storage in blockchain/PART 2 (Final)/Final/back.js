
// const express = require('express');
// const bodyParser = require('body-parser');
// const cors = require('cors');

// const app = express();
// const port = 5000;

// app.use(cors());
// app.use(bodyParser.json());

// app.post('/submit-form', (req, res) => {
//   const inputData = req.body.inputData;
//   console.log('Received input data:', inputData);
//   res.send({ message: 'Form data received successfully' });
// });

// app.listen(port, () => {
//   console.log(`Server is running on http://localhost:${port}`);
// });


const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');

const app = express();
const port = 5000;

app.use(cors());
app.use(bodyParser.json());

app.post('/submit-form', (req, res) => {
  const inputData = req.body;
  console.log('Received input data:', inputData);
  // Optionally convert to integers again if needed, for safety
  const integerData = Object.fromEntries(
    Object.entries(inputData).map(([key, value]) => [key, Number.parseInt(value, 10)])
  );
  
  const data = [[Number.parseInt(integerData[0], 10), 1, 1, 0, 0, 0, 0, 1, 0, 36, 790, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 74, 0, 40, 0, 1460, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]];
  // const data = [[integerData[0], integerData[1], integerData[2],integerData[3],integerData[4],integerData[5],integerData[6],integerData[7],integerData[8],integerData[9],integerData[10],integerData[11],integerData[12],integerData[13],integerData[14],  0, 1, 1, 1, 1, 0, 0, 0, 0, 74, 0, 40, 0, 1460, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
  const { spawn } = require('child_process');
  // Spawn a child process to run the Python script
  const pythonProcess = spawn('python', ['model.py']);
  pythonProcess.stdin.write(JSON.stringify(data));
  pythonProcess.stdin.end();
  pythonProcess.stdout.on('data', (data) => {
    const result = JSON.parse(data.toString());
    console.log('Result from Python script:', result, 'Fraud');
    let msg = ""
    if(result == 1){
      msg = "Fraud"
    }
    else{
      msg = "Not fraud"
    }

    res.send({ message: '', extraData: msg });
  });
  pythonProcess.stderr.on('data', (data) => {
    console.error(`Python script error: ${data}`);
  });

  console.log('Converted integer data:', integerData);
  // res.send({ message: 'Form data received and converted successfully' });
});

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
