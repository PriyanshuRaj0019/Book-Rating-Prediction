const predictionForm = document.getElementById("prediction-form");
const predictButton = document.getElementById("predict-button");
const exampleButton = document.getElementById("example-button");
const resetButton = document.getElementById("reset-button");

const resultEmpty = document.getElementById("result-empty");
const resultContent = document.getElementById("result-content");
const resultSubtitle = document.getElementById("result-subtitle");
const resultState = document.getElementById("result-state");
const errorBox = document.getElementById("error-box");

const predictionLabel = document.getElementById("prediction-label");
const confidenceValue = document.getElementById("confidence-value");
const confidenceBar = document.getElementById("confidence-bar");
const predictionExplanation = document.getElementById("prediction-explanation");
const modelVersion = document.getElementById("model-version");
const predictionStatus = document.getElementById("prediction-status");

const fields = {
  bookTitle: {
    input: document.getElementById("book-title"),
    error: document.getElementById("book-title-error"),
    label: "Book title",
  },
  bookCategory: {
    input: document.getElementById("book-category"),
    error: document.getElementById("book-category-error"),
    label: "Category",
  },
  priceGbp: {
    input: document.getElementById("price-gbp"),
    error: document.getElementById("price-gbp-error"),
    label: "Price",
  },
  availabilityStatus: {
    input: document.getElementById("availability-status"),
    error: document.getElementById("availability-status-error"),
    label: "Availability status",
  },
};

const exampleInput = {
  book_title: "A Light in the Attic",
  book_category: "Books",
  price_gbp: 51.77,
  availability_status: "In stock",
};

function clearFieldErrors() {
  Object.values(fields).forEach(({ input, error }) => {
    input.classList.remove("input-error");
    error.textContent = "";
  });
}

function setFieldError(field, message) {
  field.input.classList.add("input-error");
  field.error.textContent = message;
}

function getTrimmedValue(input) {
  return input.value.trim().replace(/\s+/g, " ");
}

function validateForm() {
  clearFieldErrors();

  const bookTitle = getTrimmedValue(fields.bookTitle.input);
  const bookCategory = getTrimmedValue(fields.bookCategory.input);
  const availabilityStatus = getTrimmedValue(fields.availabilityStatus.input);
  const priceGbp = Number(fields.priceGbp.input.value);

  let isValid = true;

  if (!bookTitle) {
    setFieldError(fields.bookTitle, "Book title is required.");
    isValid = false;
  }

  if (bookTitle.length > 300) {
    setFieldError(fields.bookTitle, "Book title must be 300 characters or less.");
    isValid = false;
  }

  if (!bookCategory) {
    setFieldError(fields.bookCategory, "Category is required.");
    isValid = false;
  }

  if (bookCategory.length > 100) {
    setFieldError(fields.bookCategory, "Category must be 100 characters or less.");
    isValid = false;
  }

  if (!Number.isFinite(priceGbp)) {
    setFieldError(fields.priceGbp, "Enter a valid price.");
    isValid = false;
  } else if (priceGbp <= 0) {
    setFieldError(fields.priceGbp, "Price must be greater than 0.");
    isValid = false;
  } else if (priceGbp > 1000) {
    setFieldError(fields.priceGbp, "Price must be 1000 or less.");
    isValid = false;
  }

  if (!availabilityStatus) {
    setFieldError(fields.availabilityStatus, "Availability status is required.");
    isValid = false;
  }

  if (availabilityStatus.length > 150) {
    setFieldError(fields.availabilityStatus, "Availability status must be 150 characters or less.");
    isValid = false;
  }

  if (!isValid) {
    return null;
  }

  return {
    book_title: bookTitle,
    book_category: bookCategory,
    price_gbp: priceGbp,
    availability_status: availabilityStatus,
  };
}

function setLoadingState(isLoading) {
  predictButton.disabled = isLoading;
  exampleButton.disabled = isLoading;
  resetButton.disabled = isLoading;

  if (isLoading) {
    predictButton.textContent = "Predicting...";
    resultState.textContent = "Running";
    resultSubtitle.textContent = "Sending input to the FastAPI model service.";
    errorBox.classList.add("hidden");
  } else {
    predictButton.textContent = "Predict Rating";
  }
}

function formatPredictionLabel(rawLabel) {
  if (rawLabel === "high_rating") {
    return "High Rating";
  }

  if (rawLabel === "low_rating") {
    return "Low Rating";
  }

  return "Unknown";
}

function formatProbability(probability) {
  if (probability === null || probability === undefined || Number.isNaN(Number(probability))) {
    return null;
  }

  return Math.max(0, Math.min(100, Number(probability) * 100));
}

function showError(message) {
  resultEmpty.classList.remove("hidden");
  resultContent.classList.add("hidden");
  errorBox.classList.remove("hidden");

  errorBox.textContent = message;
  resultState.textContent = "Error";
  resultSubtitle.textContent = "Prediction failed. Check the input or API status.";
}

function showPredictionResult(data) {
  const readableLabel = formatPredictionLabel(data.prediction_label);
  const highRatingProbability = formatProbability(data.high_rating_probability);
  const isHighRating = data.prediction_label === "high_rating";

  resultEmpty.classList.add("hidden");
  resultContent.classList.remove("hidden");
  errorBox.classList.add("hidden");

  predictionLabel.textContent = readableLabel;
  predictionLabel.classList.toggle("label-high", isHighRating);
  predictionLabel.classList.toggle("label-low", !isHighRating);

  if (highRatingProbability === null) {
    confidenceValue.textContent = "Not available";
    confidenceBar.style.width = "0%";
  } else {
    confidenceValue.textContent = `${highRatingProbability.toFixed(1)}%`;
    confidenceBar.style.width = `${highRatingProbability}%`;
  }

  predictionExplanation.textContent = isHighRating
    ? "The model predicts a high-rating outcome based on the provided title, category, price, and availability patterns."
    : "The model predicts a low-rating outcome based on the provided title, category, price, and availability patterns.";

  modelVersion.textContent = data.model_version || "v1.0.0";
  predictionStatus.textContent = "Completed";
  resultState.textContent = "Success";
  resultSubtitle.textContent = "Prediction completed using the deployed model bundle.";
}

async function submitPrediction(event) {
  event.preventDefault();

  const payload = validateForm();

  if (!payload) {
    showError("Please fix the highlighted fields before submitting.");
    return;
  }

  setLoadingState(true);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const responseData = await response.json();

    if (!response.ok) {
      const detail = typeof responseData.detail === "string"
        ? responseData.detail
        : "Prediction request failed.";

      throw new Error(detail);
    }

    showPredictionResult(responseData);
  } catch (error) {
    showError(error.message || "Something went wrong while calling the prediction API.");
  } finally {
    setLoadingState(false);
  }
}

function fillExampleInput() {
  fields.bookTitle.input.value = exampleInput.book_title;
  fields.bookCategory.input.value = exampleInput.book_category;
  fields.priceGbp.input.value = exampleInput.price_gbp;
  fields.availabilityStatus.input.value = exampleInput.availability_status;

  clearFieldErrors();
  errorBox.classList.add("hidden");
}

function resetResultPanel() {
  clearFieldErrors();

  resultEmpty.classList.remove("hidden");
  resultContent.classList.add("hidden");
  errorBox.classList.add("hidden");

  resultState.textContent = "Waiting";
  resultSubtitle.textContent = "Enter book details and click Predict Rating.";

  predictionLabel.textContent = "—";
  predictionLabel.classList.remove("label-high", "label-low");
  confidenceValue.textContent = "—";
  confidenceBar.style.width = "0%";
  predictionExplanation.textContent = "Based on title-derived features, category, price, and availability patterns.";
  modelVersion.textContent = "—";
  predictionStatus.textContent = "Waiting";
}

predictionForm.addEventListener("submit", submitPrediction);
exampleButton.addEventListener("click", fillExampleInput);
resetButton.addEventListener("click", resetResultPanel);

Object.values(fields).forEach(({ input }) => {
  input.addEventListener("input", () => {
    input.classList.remove("input-error");
    const errorElement = document.getElementById(`${input.id}-error`);
    if (errorElement) {
      errorElement.textContent = "";
    }
  });
});

