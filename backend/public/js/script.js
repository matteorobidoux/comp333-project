// Updates the form based on the selected input type
function updateForm() {
	const inputType = document.getElementById("inputType").value;
	const formContainer = document.getElementById("formContainer");
	const resultContainer = document.getElementById("result");

	// Clear previous form content and results
	formContainer.innerHTML = "";
	resultContainer.innerHTML = "";

	// Create form inputs based on the selected type
	if (inputType === "sms") {
		formContainer.innerHTML = `
        <div class="input-group">
            <label for="smsText">SMS Text</label>
            <textarea id="smsText" rows="4"></textarea>
        </div>
    `;
	} else if (inputType === "email") {
		formContainer.innerHTML = `
        <div class="input-group">
            <label for="emailSubject">Email Subject</label>
            <input type="text" id="emailSubject">
        </div>
        <div class="input-group">
            <label for="emailBody">Email Body</label>
            <textarea id="emailBody" rows="4"></textarea>
        </div>
    `;
	} else if (inputType === "youtube") {
		formContainer.innerHTML = `
        <div class="input-group">
            <label for="youtubeAuthor">Author</label>
            <input type="text" id="youtubeAuthor">
        </div>
        <div class="input-group">
            <label for="youtubeText">Comment Text</label>
            <textarea id="youtubeText" rows="4"></textarea>
        </div>
    `;
	}
}

// Returns the appropriate color based on the probability value
function getColor(probability) {
	let confidenceColor;
	if (probability >= 0.8) {
		confidenceColor = "#dc3545"; // red
	} else if (probability >= 0.6) {
		confidenceColor = "#ff6b35"; // orange-red
	} else if (probability >= 0.4) {
		confidenceColor = "#fdca40"; // yellow-orange
	} else if (probability >= 0.2) {
		confidenceColor = "#a7c4bc"; // muted green
	} else {
		confidenceColor = "#28a745"; // green
	}
	return confidenceColor;
}

// Sends input data to the backend and checks for spam
async function checkSpam() {
	// Clear previous results
	document.getElementById("result").innerHTML = "";

	// Get selected input type and spinner
	const inputType = document.getElementById("inputType").value;
	const spinner = document.getElementById("spinner");
	let endpoint = "";
	let body = {};
	let vercelUrl = "https://spam-detection-model.vercel.app/";

	// Set API endpoint and request body based on input type
	if (inputType === "sms") {
		const smsText = document.getElementById("smsText").value;
		endpoint = vercelUrl + `predict-sms`;
		body = { text: smsText };
	} else if (inputType === "email") {
		const emailSubject = document.getElementById("emailSubject").value;
		const emailBody = document.getElementById("emailBody").value;
		endpoint = vercelUrl + `predict-email`;
		body = { text: emailBody, subject: emailSubject };
	} else if (inputType === "youtube") {
		const youtubeAuthor = document.getElementById("youtubeAuthor").value;
		const youtubeText = document.getElementById("youtubeText").value;
		endpoint = vercelUrl + `predict-comment`;
		body = { text: youtubeText, author: youtubeAuthor };
	}

	try {
		// Show spinner while waiting for the response
		spinner.style.display = "block";

		// Send the request to the backend
		const response = await fetch(endpoint, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify(body)
		});

		// Handle response errors
		if (!response.ok) {
			throw new Error(`HTTP error! Status: ${response.status}`);
		}

		// Parse the result from the response
		const result = await response.json();
		result.prediction = JSON.parse(result.prediction);

		// Extract prediction details
		const { is_spam, combined_prob, spam_prob, url_spam_prob } =
			result.prediction;
		const percentage = (spam_prob * 100).toFixed(2);
		const combinedPercentage = (combined_prob * 100).toFixed(2);

		// Set label based on the input type
		let label = "";
		if (inputType === "sms") {
			label = "SMS Spam Probability";
		} else if (inputType === "email") {
			label = "Email Spam Probability";
		} else if (inputType === "youtube") {
			label = "Comment Spam Probability";
		}

		// Check if URL probability exists and prepare the analysis
		let urlAnalysis = "";
		if (url_spam_prob !== 0.0) {
			const urlPercentage = (url_spam_prob * 100).toFixed(2);
			urlAnalysis = `
            <div class="url-analysis">
                <p><strong>URL Spam Risk:</strong> <span style="color: ${getColor(
									url_spam_prob
								)};">${urlPercentage}%</span></p>
            </div>`;
		}

		// Build the result content to be displayed
		let formattedResult = `
        <div class="result-container">
            <p><strong>${label}:</strong> <span style="color: ${getColor(
			spam_prob
		)}; font-weight: bold;">${percentage}%</span></p>
            ${urlAnalysis}
            <div class="confidence-bar" style="width: 100%; background-color: #f0f0f0; border-radius: 5px; margin: 10px 0;">
                <div style="width: ${combinedPercentage}%; height: 20px; background-color: ${getColor(
			combined_prob
		)}; border-radius: 5px;"></div>
            </div>
            <p> <strong> Final Prediction: </strong> ${
							is_spam
								? '<span class="spam-yes">Likely Spam ' +
								  `(${combinedPercentage}% Confidence)</span>`
								: '<span class="spam-no">Likely Not Spam' +
								  ` (${100 - combinedPercentage}% Confidence)</span>`
						}</p>
        </div>`;

		// Display the result
		document.getElementById("result").innerHTML = formattedResult;
	} catch (error) {
		// Show error message if something goes wrong
		console.error("Error:", error);
		document.getElementById("result").textContent = "Error checking spam.";
	} finally {
		// Hide spinner when done
		spinner.style.display = "none";
	}
}

// Initialize the form with SMS input by default
updateForm();
