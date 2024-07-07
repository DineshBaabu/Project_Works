const express = require('express')
const request = require('request')

app = express();
const port = 3000;

app.get('/home', function(req, res){
    request('http://127.0.0.1:5000/flask', function (error, response, body) {
        console.error('error:', error);
        console.log('statusCode:', response && response.statusCode)
        console.log('body:', body);
        res.send(body);
    });
});

app.listen(port, function () {
    console.log('Listening on port 3000');                                  
})



// const axios = require('axios');

// axios.post('http://localhost:5000/flask-endpoint', { data: 'Hello from Node.js' })
//     .then(response => {
//         console.log(response.data);
//     })
//     .catch(error => {
//         console.error(error);
//     });


