document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('youtube-url-form');
    const urlInput = document.getElementById('youtube-url');
    const submitButton = document.getElementById('submit-button');
    const loadingSection = document.getElementById('loading');
    const errorSection = document.getElementById('error-message');
    const errorText = document.getElementById('error-text');
    const resultsSection = document.getElementById('results-section');
    const analyzeAnotherBtn = document.getElementById('analyze-another');
    
    // Video info elements
    const videoTitle = document.getElementById('video-title');
    const videoCategory = document.getElementById('video-category');
    const publishDate = document.getElementById('publish-date');
    const videoDuration = document.getElementById('video-duration');
    const daysSince = document.getElementById('days-since');
    
    // Prediction elements
    const predictionBadge = document.getElementById('prediction-badge');
    const recommendation = document.getElementById('recommendation');
    
    // Initialize event listeners
    form.addEventListener('submit', handleFormSubmit);
    analyzeAnotherBtn.addEventListener('click', resetForm);
    
    async function handleFormSubmit(event) {
        event.preventDefault();
        
        const youtubeUrl = urlInput.value.trim();
        
        if (!isValidYouTubeUrl(youtubeUrl)) {
            showError('Please enter a valid YouTube URL.');
            return;
        }
        
        // Show loading state
        showLoading(true);
        hideResults();
        hideError();
        disableSubmitButton(true);
        
        try {
            const response = await fetch('http://127.0.0.1:5000/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url: youtubeUrl })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to analyze the YouTube URL');
            }
            
            const result = await response.json();
            console.log('Analysis Result:', result);
            
            // Process and display the results
            displayResults(result);
            
        } catch (error) {
            console.error('Error:', error);
            showError(error.message);
        } finally {
            showLoading(false);
            disableSubmitButton(false);
        }
    }
    
    function displayResults(data) {
        // Display video information
        const videoInfo = data.video_info;
        videoTitle.textContent = videoInfo.title;
        videoCategory.textContent = videoInfo.category;
        
        // Format and display date
        const date = new Date(videoInfo.publish_date);
        const formattedDate = date.toLocaleDateString('en-US', { 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric' 
        });
        publishDate.textContent = formattedDate;
        
        // Format and display duration
        videoDuration.textContent = formatDuration(videoInfo.duration);
        
        // Display days since published
        daysSince.textContent = videoInfo.days_since_publish;
        
        // Display prediction results
        const prediction = data.prediction;
        
        // Set prediction badge
        predictionBadge.textContent = prediction.label;
        predictionBadge.className = 'prediction-badge'; // Reset classes
        
        // Add appropriate class based on prediction
        switch(prediction.class) {
            case 0:
                predictionBadge.classList.add('not-popular');
                break;
            case 1:
                predictionBadge.classList.add('controversy');
                break;
            case 2:
                predictionBadge.classList.add('decent');
                break;
            case 3:
                predictionBadge.classList.add('overwhelming');
                break;
        }
        
        // Set recommendation text
        recommendation.textContent = `We ${prediction.recommendation} this video.`;
        
        // Display probability bars
        prediction.probabilities.forEach(prob => {
            const index = prob.label.toLowerCase().includes('not popular') ? 0 : 
                         prob.label.toLowerCase().includes('controversy') ? 1 :
                         prob.label.toLowerCase().includes('decent') ? 2 : 3;
            
            const bar = document.getElementById(`prob-${index}`);
            const value = document.getElementById(`prob-${index}-value`);
            
            if (bar && value) {
                bar.style.width = `${prob.value}%`;
                value.textContent = `${prob.value}%`;
            }
        });
        
        // Show results section
        showResults();
    }
    
    function formatDuration(seconds) {
        if (!seconds) return "0:00";
        
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        
        if (hours > 0) {
            return `${hours}:${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
        } else {
            return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
        }
    }
    
    function showLoading(show) {
        loadingSection.classList.toggle('hidden', !show);
    }
    
    function showError(message) {
        errorText.textContent = message;
        errorSection.classList.remove('hidden');
    }
    
    function hideError() {
        errorSection.classList.add('hidden');
    }
    
    function showResults() {
        resultsSection.classList.remove('hidden');
    }
    
    function hideResults() {
        resultsSection.classList.add('hidden');
    }
    
    function resetForm() {
        urlInput.value = '';
        hideResults();
        urlInput.focus();
    }
    
    function disableSubmitButton(disable) {
        submitButton.disabled = disable;
        if (disable) {
            submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
        } else {
            submitButton.innerHTML = '<i class="fas fa-search"></i> Analyze';
        }
    }
    
    function isValidYouTubeUrl(url) {
        const regex = /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.?be)\/.+$/;
        return regex.test(url);
    }
});