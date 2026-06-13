const runBtn = document.querySelector(".nav-cta");
const fileInput = document.getElementById("imageInput");
const outputImage = document.getElementById("outputImage");
const resultBox = document.getElementById("result");

runBtn.onclick = () => fileInput.click();

fileInput.onchange = async () => {
  const file = fileInput.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  try {
    runBtn.innerText = "Processing...";

    const res = await fetch("http://127.0.0.1:8000/api/predict", {
      method: "POST",
      body: formData
    });

    // 🔥 THIS IS CRITICAL DEBUG STEP
    if (!res.ok) {
      const text = await res.text();
      console.log("Backend error:", text);
      throw new Error(text);
    }

    const data = await res.json();

    outputImage.src = "data:image/png;base64," + data.overlay_b64;
    outputImage.style.display = "block";

    resultBox.innerHTML = `
      Tumor: ${data.tumor_area_pct.toFixed(2)}% <br>
      Dice: ${data.dice} <br>
      Accuracy: ${data.accuracy} <br>
      SSIM: ${data.ssim} <br>
      PSNR: ${data.psnr}
    `;

  } catch (err) {
    console.error(err);
    resultBox.innerHTML = "Backend Error ❌ Check console (F12)";
  } finally {
    runBtn.innerText = "Run Model";
  }
};