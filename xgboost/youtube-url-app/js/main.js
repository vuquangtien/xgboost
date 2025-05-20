document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('youtube-url-form');
    const urlInput = document.getElementById('youtube-url');

    form.addEventListener('submit', async function(event) {
        event.preventDefault();

        const youtubeUrl = urlInput.value;

        if (!isValidYouTubeUrl(youtubeUrl)) {
            alert('Please enter a valid YouTube URL.');
            return;
        }

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
                throw new Error(errorData.error || 'Failed to analyze the YouTube URL.');
            }

            const result = await response.json();
            console.log('Analysis Result:', result);
            alert(`Analysis Result: ${JSON.stringify(result, null, 2)}`);
        } catch (error) {
            console.error('Error:', error);
            alert(`An error occurred: ${error.message}`);
        }

        urlInput.value = '';
    });

    function isValidYouTubeUrl(url) {
        const regex = /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.?be)\/.+$/;
        return regex.test(url);
    }
});