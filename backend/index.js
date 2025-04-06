const express = require("express");
const { spawn } = require("child_process");
const path = require("path");
const app = express();
const cors = require("cors");
const port = 3000;

// Middleware to parse JSON data and enable CORS
app.use(express.json());
app.use(cors());

// Serve static content from the 'public' directory
app.use(express.static(path.join(__dirname, "public")));

// Handles SMS spam prediction requests
app.post("/predict-sms", (req, res) => {
	const pythonProcess = spawn("python", [
		"models/sms/predict_sms_spam.py",
		req.body.text
	]);

	pythonProcess.stdout.on("data", (data) => {
		const prediction = data.toString().trim();
		console.log(`Prediction: ${prediction}`);
		res.json({ prediction });
	});

	pythonProcess.stderr.on("data", (data) => {
		console.error(`stderr: ${data}`);
		res.status(500).send("Error occurred while processing the request.");
	});

	pythonProcess.on("close", (code) => {
		if (code !== 0) {
			console.error(`Python process exited with code ${code}`);
			res.status(500).send("Error occurred while processing the request.");
		}
	});
});

// Handles Email spam prediction requests
app.post("/predict-email", (req, res) => {
	const pythonProcess = spawn("python", [
		"models/email/predict_email_spam.py",
		req.body.subject,
		req.body.text
	]);

	pythonProcess.stdout.on("data", (data) => {
		const prediction = data.toString().trim();
		console.log(`Prediction: ${prediction}`);
		res.json({ prediction });
	});

	pythonProcess.stderr.on("data", (data) => {
		console.error(`stderr: ${data}`);
		res.status(500).send("Error occurred while processing the request.");
	});

	pythonProcess.on("close", (code) => {
		if (code !== 0) {
			console.error(`Python process exited with code ${code}`);
			res.status(500).send("Error occurred while processing the request.");
		}
	});
});

// Handles YouTube comment spam prediction requests
app.post("/predict-comment", (req, res) => {
	const pythonProcess = spawn("python", [
		"models/comment/predict_comment_spam.py",
		req.body.author,
		req.body.text
	]);

	pythonProcess.stdout.on("data", (data) => {
		const prediction = data.toString().trim();
		console.log(`Prediction: ${prediction}`);
		res.json({ prediction });
	});

	pythonProcess.stderr.on("data", (data) => {
		console.error(`stderr: ${data}`);
		res.status(500).send("Error occurred while processing the request.");
	});

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
