// script.js

function processFile() {
    const operation = document.getElementById("operation").value;
    const fileInput = document.getElementById("fileUpload");
    const processingText = document.getElementById("processingText");

    if (fileInput.files.length === 0) {
        alert("Please upload a file.");
        return;
    }
    
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("operation", operation);

    // Show Processing Text
    processingText.style.display = "block";

    fetch("/process", {
        method: "POST",
        body: formData,
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to process the file.');
        }
        return response.blob();
    })
    .then(data => {
        const outputImage = document.getElementById("outputImage");
        const outputVideo = document.getElementById("outputVideo");
        const objectURL = URL.createObjectURL(data);
        
        // Hide Processing Text
        processingText.style.display = "none";
        
        // Check file type and display it
        if (data.type.includes("image")) {
            outputImage.src = objectURL;
            outputImage.style.display = "block";
            outputVideo.style.display = "none";
        } else if (data.type.includes("video")) {
            outputVideo.src = objectURL;
            outputVideo.style.display = "block";
            outputImage.style.display = "none";
        }
    })
    .catch(error => {
        alert("Error processing the file.");
        console.error(error);
        // Hide Processing Text in case of error
        processingText.style.display = "none";
    });
}

function downloadOutput() {
    fetch("/download")
        .then(response => response.blob())
        .then(data => {
            const link = document.createElement("a");
            link.href = URL.createObjectURL(data);
            link.download = "output.mp4";
            link.click();
        })
        .catch(error => {
            alert("No processed file available.");
            console.error(error);
        });
}
