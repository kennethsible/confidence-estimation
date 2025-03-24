let confidenceThreshold = 8.38; // 8.382382382382382
let restrictVocab = null;
let ntotal = null;

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
    if (!inputElement.textContent.trim()) {
        alert('You must enter a sentence in the input field (top/left).');
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
            const buttonElement = document.getElementById('buttonIcon');
            buttonElement.className = 'fas fa-circle-exclamation';
            console.error(error);
        }
    );
}

function handleWordClick(event) {
    const contextMenu = document.getElementById('context-menu');
    const hideContextMenu = () => {
        contextMenu.style.display = 'none';
        restrictVocab = null;
    };

    const menuItemsContainer = document.getElementById('menu-items');
    event.target.style.cursor = 'wait';
    callNeighborsFunction(event.target.textContent.trim()).then(
        function (response) {
            event.target.style = '';

            menuItemsContainer.innerHTML = response['neighbors'].map(item => `<li>${item}</li>`).join('');

            const refineSearchItem = document.createElement('li');
            refineSearchItem.textContent = 'Narrow Search';
            refineSearchItem.style.fontWeight = 'bold';
            refineSearchItem.style.border = '1px solid #ccc';
            refineSearchItem.style.borderRadius = '6px';

            menuItemsContainer.appendChild(refineSearchItem);

            menuItemsContainer.querySelectorAll('li:not(:last-child)').forEach(li => {
                li.addEventListener('click', (e) => {
                    hideContextMenu();
                    event.target.textContent = e.target.textContent.trim();
                    event.target.className = '';
                    event.target.style = '';
                    event.target.removeEventListener('click', handleWordClick);
                });
            });

            refineSearchItem.addEventListener('click', () => {
                if (restrictVocab === null) {
                    restrictVocab = ntotal;
                }
                restrictVocab = Math.floor(restrictVocab / 2);
                if (restrictVocab > 0) {
                    handleWordClick(event);
                }
                else {
                    refineSearchItem.style.pointerEvents = 'none';
                    refineSearchItem.style.cursor = 'not-allowed';
                    refineSearchItem.style.color = '#999';
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

async function callNtotalFunction() {
    const apiUrl = 'http://localhost:8080/ntotal';

    const response = await fetch(apiUrl);
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
    if (event.key === 'Enter') {
        const activeElement = document.activeElement;
        if (activeElement.isContentEditable) {
            event.preventDefault();
        }
        highlightWords();
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

    callNtotalFunction().then(
        function (response) {
            ntotal = response['ntotal'];
        },
        function (error) {
            console.error(error);
        }
    );
});
