// Get elements from the DOM
let userInput = document.getElementById('user-input');
let chatBox = document.getElementById('chat-box');

// Function to check if Enter key is pressed
function checkEnter(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

// Function to send a message
function sendMessage() {
    const message = userInput.value.trim();
    if (message) {
        addMessageToChatBox(message, 'user'); // Show user message
        userInput.value = ''; // Clear input field

        // Send the message to the server
        fetch('/get_response', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            addMessageToChatBox(data.response, 'bot'); // Show bot response
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
}

// Function to add message to chat box with typing animation
function addMessageToChatBox(message, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add(sender + '-message');
    const innerMessageDiv = document.createElement('div');
    innerMessageDiv.classList.add('message');
    innerMessageDiv.setAttribute('data-sender', sender === 'user' ? 'You' : 'Bot');
    messageDiv.appendChild(innerMessageDiv);
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll to bottom

    let index = 0;
    const typingSpeed = 50;

    function typeMessage() {
        if (index < message.length) {
            const char = message.charAt(index);
            innerMessageDiv.innerHTML += char === ' ' ? '&nbsp;' : char;
            index++;
            setTimeout(typeMessage, typingSpeed);
        }
    }

    if (sender === 'bot') {
        setTimeout(typeMessage, 500);
    } else {
        innerMessageDiv.innerText = message;
    }
}

// Navigation function to prevent multiple clicks
function navigateOnce(url) {
    document.getElementById('login-btn').disabled = true;
    document.getElementById('signup-btn').disabled = true;
    window.location.href = url;
}

// Automatically greet the user on page load
window.onload = function() {
    addMessageToChatBox("Welcome to Chatterbot!", 'bot');
};

