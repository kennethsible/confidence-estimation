let confidenceThreshold = 8.38;
let topKNeighbors = 5;
let restrictVocab = null;

function toggleEditable() {
    const inputElement = document.getElementById('inputText');
    const iconElement = document.querySelector('.lock-icon');
    const iconSymbol = document.querySelector('.lock-icon i');
    if (inputElement.getAttribute('contenteditable') === 'false') {
        inputElement.innerHTML = inputElement.textContent;
        inputElement.setAttribute('contenteditable', 'true');
        iconElement.style.pointerEvents = 'none';
        iconSymbol.className = 'fa-solid fa-lock-open';
    }
}

function toggleClickable() {
    const inputElement = document.getElementById('inputText');
    inputElement.querySelectorAll('span.clickable').forEach(span => {
        span.removeEventListener('click', handleWordClick);
    });
    inputElement.querySelectorAll('span').forEach(span => {
        span.classList.add('clickable');
    });
    inputElement.querySelectorAll('span.clickable').forEach(span => {
        span.addEventListener('click', handleWordClick);
    });
}

function highlightWords() {
    const inputElement = document.getElementById('inputText');
    const buttonElement = document.getElementById('buttonIcon');
    if (buttonElement.classList.contains('fa-circle-exclamation')) {
        alert('The API server is currently unreachable.');
        return;
    } else if (!inputElement.textContent.trim()) {
        alert('You must first enter an input sentence.');
        return;
    }

    callTranslateFunction().then(
        function (response) {
            const inputElement = document.getElementById('inputText');
            inputElement.innerHTML = inputElement.textContent;
            inputElement.setAttribute('contenteditable', 'false');
            const iconElement = document.querySelector('.lock-icon');
            iconElement.style.pointerEvents = 'auto';
            const iconSymbol = document.querySelector('.lock-icon i');
            iconSymbol.className = 'fa-solid fa-lock';
            const plainText = inputElement.innerHTML;

            const wordsToHighlight = response['scores'].filter(pair => pair[1] > confidenceThreshold).map(pair => pair[0]);
            const escapeRegExp = (string) => string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            if (wordsToHighlight.length > 0) {
                const regex = new RegExp(`\\b(${wordsToHighlight.map(escapeRegExp).join('|')})\\b`, 'gi');
                inputElement.innerHTML = plainText.replace(regex, match => {
                    return `<span class="highlight">${match}</span>`;
                });
            } else {
                inputElement.innerHTML = plainText;
            }

            toggleClickable();
        },
        function (error) {
            buttonElement.className = 'fas fa-circle-exclamation';
            alert('The API server is currently unreachable.');
            console.error(error);
        }
    );
}

function handleWordClick(event) {
    const contextMenu = document.getElementById('context-menu');
    const hideContextMenu = () => {
        contextMenu.style.display = 'none';
    };

    const menuItemsContainer = document.getElementById('menu-items');
    event.target.style.cursor = 'wait';
    callNeighborsFunction(event.target.textContent.trim()).then(
        function (response) {
            event.target.style.cursor = '';

            menuItemsContainer.innerHTML = response['neighbors'].map(item => `<li>${item}</li>`).join('');

            const contextInput = document.createElement('li');
            const inputElement = document.createElement('input');

            inputElement.type = 'text';
            inputElement.placeholder = 'Enter Substitute';
            inputElement.setAttribute('autocapitalize', 'off');
            inputElement.classList.add('input-element');
            contextInput.classList.add('context-input');
            contextInput.appendChild(inputElement);

            menuItemsContainer.appendChild(contextInput);

            if (window.matchMedia('(orientation: landscape)').matches) {
                setTimeout(() => inputElement.focus(), 0);
            }

            menuItemsContainer.querySelectorAll('li:not(:last-child)').forEach(li => {
                li.addEventListener('click', (e) => {
                    hideContextMenu();
                    event.target.textContent = e.target.textContent.trim();
                    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
                        event.target.style.backgroundColor = '#525252';
                    } else {
                        event.target.style.backgroundColor = '#a5a5a5';
                    }
                });
            });

            inputElement.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && inputElement.value.trim() !== '') {
                    hideContextMenu();
                    event.target.textContent = inputElement.value.trim();
                    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
                        event.target.style.backgroundColor = '#525252';
                    } else {
                        event.target.style.backgroundColor = '#a5a5a5';
                    }
                }
            });

            contextMenu.style.display = 'block';
            const menuWidth = contextMenu.offsetWidth;
            const screenWidth = window.innerWidth;
            const clickX = event.pageX;
            if (clickX + menuWidth > screenWidth) {
                contextMenu.style.left = `${clickX - menuWidth}px`;
            } else {
                contextMenu.style.left = `${clickX}px`;
            }
            contextMenu.style.top = `${event.pageY}px`;
        },
        function (error) {
            event.target.style.cursor = 'default';
            console.error(error);
        }
    );

    document.addEventListener('click', (event) => {
        const insideMenu = contextMenu.contains(event.target);
        const onHighlight = event.target.classList.contains('highlight');
        if (!insideMenu && !onHighlight) { hideContextMenu(); }
    });
}

async function checkAPIHealth() {
    const apiUrl = 'http://localhost:8080/health';
    const buttonElement = document.getElementById('buttonIcon');

    try {
        const response = await fetch(apiUrl);
        if (!response.ok) {
            buttonElement.className = 'fas fa-circle-exclamation';
            throw new Error('Request Failed.',);
        }
    } catch (error) {
        buttonElement.className = 'fas fa-circle-exclamation';
        console.error(error);
    }
}

async function callTranslateFunction() {
    const apiUrl = 'http://localhost:8080/translate';
    const inputElement = document.getElementById('inputText');
    const outputElement = document.getElementById('outputText');
    const buttonElement = document.getElementById('buttonIcon');
    const checkbox = document.getElementById('send-data');

    const className = buttonElement.className;
    buttonElement.className = 'fas fa-sync fa-spin';

    const requestData = {
        string: inputElement.textContent,
        send_data: checkbox.checked,
    };
    const requestOptions = {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
    };

    const response = await fetch(apiUrl, requestOptions);
    if (!response.ok) {
        throw new Error('Request Failed.');
    }
    const json = await response.json();
    console.log(json['scores']);
    outputElement.textContent = json['output'];
    buttonElement.className = className;
    return json;
}

async function callNeighborsFunction(word) {
    const apiUrl = 'http://localhost:8080/neighbors';
    const checkbox = document.getElementById('send-data');

    const requestData = {
        string: word,
        n_neighbors: topKNeighbors,
        restrict_vocab: restrictVocab,
        send_data: checkbox.checked,
    };
    const requestOptions = {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
    };

    const response = await fetch(apiUrl, requestOptions);
    if (!response.ok) {
        throw new Error('Request Failed.');
    }
    return await response.json();
}

const inputText = document.getElementById('inputText');

inputText.addEventListener('paste', function (e) {
    e.preventDefault();
    const plainText = e.clipboardData.getData('text/plain');
    const selection = window.getSelection();
    const range = selection.getRangeAt(0);

    const textNode = document.createTextNode(plainText);
    range.deleteContents();
    range.insertNode(textNode);

    range.setStartAfter(textNode);
    range.setEndAfter(textNode);
    selection.removeAllRanges();
    selection.addRange(range);
});

document.addEventListener('keydown', function (event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        const activeElement = document.activeElement;
        if (activeElement.isContentEditable) {
            event.preventDefault();
            highlightWords();
        }
    }
});

function shakeLockIcon() {
    const inputElement = document.getElementById('inputText');
    const lockIcon = document.querySelector('.lock-icon');
    if (inputElement.getAttribute('contenteditable') === 'false') {
        lockIcon.classList.add('shake');
        setTimeout(() => {
            lockIcon.classList.remove('shake');
        }, 500);
    }
}

function updateCollapsibleWidth() {
    const collapsible = document.querySelector('.collapsible');
    const container = document.querySelector('.container');
    if (collapsible && container) {
        collapsible.style.width = `${container.offsetWidth * 0.8}px`;
    }
}

function updateLockIconPosition() {
    let inputText = document.getElementById('inputText');
    let lockIcon = document.querySelector('.lock-icon');

    if (inputText && lockIcon) {
        let rect = inputText.getBoundingClientRect();

        let positionTop = rect.bottom - lockIcon.offsetHeight - 10;
        let positionLeft = rect.right - lockIcon.offsetWidth - 10;

        lockIcon.style.top = `${positionTop}px`;
        lockIcon.style.left = `${positionLeft}px`;
        lockIcon.style.visibility = 'visible';
    }
}

function updateLayout() {
    updateCollapsibleWidth();
    updateLockIconPosition();
}

window.addEventListener('resize', updateLayout);
window.addEventListener('load', updateLayout);

document.addEventListener('DOMContentLoaded', function () {
    const details = document.getElementById('info-details');
    const scrollIndicator = document.getElementById('scroll-indicator');

    function updateScrollIndicator() {
        if (!details.open) {
            scrollIndicator.style.transition = 'none';
            scrollIndicator.classList.remove('visible');
            return;
        }
        setTimeout(() => {
            scrollIndicator.style.transition = 'opacity 0.3s ease-in-out';
        }, 10);
        if (details.scrollHeight > details.clientHeight) {
            if (details.scrollTop + details.clientHeight >= details.scrollHeight - 5) {
                scrollIndicator.classList.remove('visible');
            } else {
                scrollIndicator.classList.add('visible');
            }
        } else {
            scrollIndicator.classList.remove('visible');
        }
    }

    details.addEventListener('scroll', updateScrollIndicator);
    details.addEventListener('toggle', updateScrollIndicator);
    details.addEventListener('toggle', updateLockIconPosition);

    scrollIndicator.addEventListener('click', function () {
        details.scrollTo({ top: details.scrollHeight, behavior: 'smooth' });
    });

    updateScrollIndicator();

    const checkbox = document.getElementById('send-data');
    const savedState = localStorage.getItem('sendTranslationData');
    if (savedState !== null) {
        checkbox.checked = JSON.parse(savedState);
    }
    checkbox.addEventListener('change', function () {
        localStorage.setItem('sendTranslationData', checkbox.checked);
    });

    checkAPIHealth();
});
