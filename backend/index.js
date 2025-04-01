const express = require("express");
const { spawn } = require("child_process");
const path = require("path");
const app = express();
const cors = require("cors");
const port = 3000;

// Middleware to parse JSON data and enable CORS
app.use(express.json());
app.use(cors());

// Handles SMS spam prediction requests
app.post("/predict-sms", (req, res) => {
	// Spawn a Python process to run the SMS spam prediction script
	const pythonProcess = spawn("python", [
		"models/sms/predict_sms_spam.py",
		req.body.text
	]);

	// Handle data returned from the Python script
	pythonProcess.stdout.on("data", (data) => {
		const prediction = data.toString().trim();
		console.log(`Prediction: ${prediction}`);
		res.json({ prediction });
	});

	// Handle errors from the Python script
	pythonProcess.stderr.on("data", (data) => {
		console.error(`stderr: ${data}`);
		res.status(500).send("Error occurred while processing the request.");
	});

	// Handle when the Python process finishes
	pythonProcess.on("close", (code) => {
		if (code !== 0) {
			console.error(`Python process exited with code ${code}`);
			res.status(500).send("Error occurred while processing the request.");
		}
	});
});

// Handles Email spam prediction requests
app.post("/predict-email", (req, res) => {
	console.log(req.body.text);
	console.log(req.body.subject);

	// Spawn a Python process to run the Email spam prediction script
	const pythonProcess = spawn("python", [
		"models/email/predict_email_spam.py",
		req.body.subject,
		req.body.text
	]);

	// Handle data returned from the Python script
	pythonProcess.stdout.on("data", (data) => {
		const prediction = data.toString().trim();
		console.log(`Prediction: ${prediction}`);
		res.json({ prediction });
	});

	// Handle errors from the Python script
	pythonProcess.stderr.on("data", (data) => {
		console.error(`stderr: ${data}`);
		res.status(500).send("Error occurred while processing the request.");
	});

	// Handle when the Python process finishes
	pythonProcess.on("close", (code) => {
		if (code !== 0) {
			console.error(`Python process exited with code ${code}`);
			res.status(500).send("Error occurred while processing the request.");
		}
	});
});

// Handles YouTube comment spam prediction requests
app.post("/predict-comment", (req, res) => {
	// Spawn a Python process to run the Comment spam prediction script
	const pythonProcess = spawn("python", [
		"models/comment/predict_comment_spam.py",
		req.body.author,
		req.body.text
	]);

	// Handle data returned from the Python script
	pythonProcess.stdout.on("data", (data) => {
		const prediction = data.toString().trim();
		console.log(`Prediction: ${prediction}`);
		res.json({ prediction });
	});

	// Handle errors from the Python script
	pythonProcess.stderr.on("data", (data) => {
		console.error(`stderr: ${data}`);
		res.status(500).send("Error occurred while processing the request.");
	});

	// Handle when the Python process finishes
	pythonProcess.on("close", (code) => {
		if (code !== 0) {
			console.error(`Python process exited with code ${code}`);
			res.status(500).send("Error occurred while processing the request.");
		}
	});
});

// Start the Express server and listen on the specified port
app.listen(port, () => {
	console.log(`Listening on port ${port}`);
});
