// React frontend (App.js)

// import React, { useState } from 'react';
// import axios from 'axios';

// function App() {
//   const [inputData, setInputData] = useState('');
//   const [responseMsg, setResponseMsg] = useState('');

//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     try {
//       const response = await axios.post('http://localhost:5000/submit-form', {
//         inputData: inputData
//       });
//       setResponseMsg(response.data.message);
//     } catch (error) {
//       console.error('Error submitting form:', error);
//     }
//   };

//   return (
//     <div>
//       <form onSubmit={handleSubmit}>
//         <input type="text" value={inputData} onChange={(e) => setInputData(e.target.value)} />
//         <button type="submit">Submit</button>
//       </form>
//       <p>Response from server: {responseMsg}</p>
//     </div>
//   );
// }

// export default App;


// import React, { useState } from "react";

// const Form = () => {
//   const [formData, setFormData] = useState({
//     Age: "",
//     Country_0: "",
//     OPTotalAmount: "",
//     TotalClaimAmount: "",
//     State_0: "",
//     IPTotalAmount: "",
//     Country_1: "",
//     Features: "",
//     State_1: "",
//     ClaimSettlementDelay: "",
//     TreatmentDuration: "",
//     ChronicCond_Depression: false,
//     ChronicCond_Osteoporasis: false,
//     Race: "",
//     ChronicCond_Alzheimer: false,
//     Gender: ""
//   });

//   const handleChange = (e) => {
//     const { name, value, type, checked } = e.target;
//     setFormData((prevData) => ({
//       ...prevData,
//       [name]: type === "checkbox" ? checked : value
//     }));
//   };

//   const handleSubmit = (e) => {
//     e.preventDefault();
//     console.log(formData);
//   };

//   return (
//     <div style={{ textAlign: "center" }}>
//     <h1>Insurance Fraud Detection</h1>
//     <div style={{marginBottom: "10px"}}></div>
//     <form onSubmit={handleSubmit}>
//         <div style={{ textAlign: "left", display: "inline-block" }}>
          

//         <div style={{marginBottom: "20px"}}>
//         <label style={{ marginRight: "300px" }}>
//           Age
//           <input
//             type="number"
//             name="Age"
//             value={formData.Age}
//             onChange={handleChange}
//           />
//         </label>
//       </div>
      
//       <div style={{marginBottom: "20px"}}>
//         <label>
//           Country_0
//           <input
//             type="text"
//             name="Country_0"
//             value={formData.Country_0}
//             onChange={handleChange}
//           />
//         </label>
//       </div>
      
//       <div style={{marginBottom: "20px"}}>
//         <label>
//           OPTotalAmount
//           <input
//             type="number"
//             name="OPTotalAmount"
//             value={formData.OPTotalAmount}
//             onChange={handleChange}
//           />
//         </label>
//       </div>
      
//       <div style={{marginBottom: "20px"}}>
//         <label>
//           TotalClaimAmount
//           <input
//             type="number"
//             name="TotalClaimAmount"
//             value={formData.TotalClaimAmount}
//             onChange={handleChange}
//           />
//         </label>
//       </div>
      
//       <div style={{marginBottom: "20px"}}>
//         <label>
//           State_0
//           <input
//             type="text"
//             name="State_0"
//             value={formData.State_0}
//             onChange={handleChange}
//           />
//         </label>
//       </div>
      
//       <div style={{marginBottom: "20px"}}>
//         <label>
//           IPTotalAmount
//           <input
//             type="number"
//             name="IPTotalAmount"
//             value={formData.IPTotalAmount}
//             onChange={handleChange}
//           />
//         </label>
//       </div>
      
//       <div style={{marginBottom: "20px"}}>
//         <label>
//           Country_1
//           <input
//             type="text"
//             name="Country_1"
//             value={formData.Country_1}
//             onChange={handleChange}
//           />
//         </label>
//       </div>
      
//       <div style={{marginBottom: "20px"}}>
//         <label>
//           Features
//           <input
//             type="text"
//             name="Features"
//             value={formData.Features}
//             onChange={handleChange}
//           />
//         </label>
//       </div>
      
//       <div style={{marginBottom: "20px"}}>
//         <label>
//           State_1
//           <input
//             type="text"
//             name="State_1"
//             value={formData.State_1}
//             onChange={handleChange}
//           />
//         </label>
//       </div>
      
//       <div style={{marginBottom: "20px"}}>
//         <label>
//           ClaimSettlementDelay
//           <input
//             type="number"
//             name="ClaimSettlementDelay"
//             value={formData.ClaimSettlementDelay}
//             onChange={handleChange}
//           />
//         </label>
//       </div>
      
//       <div style={{marginBottom: "20px"}}>
//         <label>
//           TreatmentDuration
//           <input
//             type="number"
//             name="TreatmentDuration"
//             value={formData.TreatmentDuration}
//             onChange={handleChange}
//           />
//         </label>
//       </div>
      
//       <div style={{marginBottom: "20px"}}>
//         <label>
//           ChronicCond_Depression
//           <input
//             type="checkbox"
//             name="ChronicCond_Depression"
//             checked={formData.ChronicCond_Depression}
//             onChange={handleChange}
//           />
//         </label>
//       </div>
      
//       <div style={{marginBottom: "20px"}}>
        
//         <label>
//           ChronicCond_Osteoporasis
//           <input
//             type="checkbox"
//             name="ChronicCond_Osteoporasis"
//             checked={formData.ChronicCond_Osteoporasis}
//             onChange={handleChange}
//           />
//         </label>
//       </div>
      
//       <div style={{marginBottom: "20px"}}>
//         <label>
//           Race
//           <input
//             type="text"
//             name="Race"
//             value={formData.Race}
//             onChange={handleChange}
//           />
//         </label>
//       </div>
      
//       <div style={{marginBottom: "20px"}}>
//         <label>
//           ChronicCond_Alzheimer
//           <input
//             type="checkbox"
//             name="ChronicCond_Alzheimer"
//             checked={formData.ChronicCond_Alzheimer}
//             onChange={handleChange}
//           />
//         </label>
//       </div>
      
//       <div>
//         <label>
//           Gender
//           <select
//             name="Gender"
//             value={formData.Gender}
//             onChange={handleChange}
//           >
//             <option value="">Select</option>
//             <option value="Male">Male</option>
//             <option value="Female">Female</option>
//             <option value="Other">Other</option>
//           </select>
//           <br/>
//           <br/>
//         </label>
//       </div>
//       </div>
    
//     </form>
//     <button type="submit" style={{padding: "10px 10px", backgroundColor: "red", color: "black", border: "none", borderRadius: "5px", fontSize: "16px"}}>Detect Fraud</button>
//     </div>
//   );
// };

// export default Form;



// import React, { useState } from 'react';
// import axios from 'axios';

// function App() {
//   const [inputs, setInputs] = useState({
//     input1: '',
//     input2: '',
//     input3: '',
//     input4: '',
//     input5: '',
//     input6: '',
//     input7: '',
//     input8: '',
//     input9: '',
//     input10: '',
//     input11: '',
//     input12: '',
//     input13: '',
//     input14: '',
//     input15: ''
//   });
//   const [responseMsg, setResponseMsg] = useState('');

//   const handleInputChange = (e) => {
//     const { name, value } = e.target;
//     setInputs(prevState => ({
//       ...prevState,
//       [name]: value
//     }));
//   };

//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     const dataToSend = Object.fromEntries(
//       Object.entries(inputs).map(([key, value]) => [key, parseInt(value, 10)])
//     );
//     try {
//       const response = await axios.post('http://localhost:5000/submit-form', dataToSend);
//       setResponseMsg(response.data.message + " " + response.data.extraData);
//     } catch (error) {
//       console.error('Error submitting form:', error);
//     }
//   };

//   return (
//     <div>
//       <form onSubmit={handleSubmit}>
//         {Object.keys(inputs).map((inputKey, index) => (
//           <input
//             key={index}
//             type="text"
//             name={inputKey}
//             value={inputs[inputKey]}
//             onChange={handleInputChange}
//           />
//         ))}
//         <button type="submit">Submit</button>
//       </form>
//       <p>Response from model: {responseMsg}</p>
//     </div>
//   );
// }

// export default App;

import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [inputs, setInputs] = useState({
    input1: '',
    input2: '',
    input3: '',
    input4: '',
    input5: '',
    input6: '',
    input7: '',
    input8: '',
    input9: '',
    input10: '',
    input11: '',
    input12: '',
    input13: '',
    input14: '',
    input15: ''
  });
  const [responseMsg, setResponseMsg] = useState('');

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setInputs(prevState => ({
      ...prevState,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const dataToSend = Object.fromEntries(
      Object.entries(inputs).map(([key, value]) => [key, parseInt(value, 10)])
    );
    try {
      const response = await axios.post('http://localhost:5000/submit-form', dataToSend);
      setResponseMsg(response.data.message + " " + response.data.extraData);
    } catch (error) {
      console.error('Error submitting form:', error);
    }
  };
  const features = [
    "Age",
    "Country_0",
    "OPTotalAmount",
    "TotalClaimAmount",
    "State_0",
    "IPTotalAmount",
    "Country_1",
    "Features",
    "State_1",
    "ClaimSettlementDelay",
    "TreatmentDuration",
    "ChronicCond_Depression",
    "ChronicCond_Osteoporasis",
    "Race",
    "ChronicCond_Alzheimer",
    "Gender"
  ];
  

  
  return (
    <div>
      <form onSubmit={handleSubmit}>
        {Object.keys(inputs).map((inputKey, index) => (
          <div key={index} style={{ marginBottom: '10px' }}>
            <input
              type="text"
              placeholder={features[index]}
              name={inputKey}
              value={inputs[inputKey]}
              onChange={handleInputChange}
              style={{ display: 'block', width: '100%' }}
            />
          </div>
        ))}
        <button type="submit">Detect Fraud</button>
      </form>
      <p>Response from model: {responseMsg}</p>
    </div>
  );
  
}

export default App;

