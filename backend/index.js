const express = require("express");
const { spawn } = require("child_process");
const path = require("path");
const app = express();
const cors = require("cors");
const port = 3000;

app.use(express.json());
app.use(cors());

// takes data from the frontend and returns a prediction
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

app.post("/predict-email", (req, res) => {
	console.log(req.body.text);
	console.log(req.body.subject);

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

app.listen(port, () => {
	console.log(`Listening on port ${port}`);
});
