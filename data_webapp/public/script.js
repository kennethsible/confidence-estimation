function toggleEditable() {
    const inputElement = document.getElementById('inputText');
    inputElement.innerHTML = inputElement.textContent;
    inputElement.setAttribute('contenteditable', true);
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
    const threshold = 8; // TODO 8.382382382382382

    callTranslateFunction().then(
        function(value) {
            const inputElement = document.getElementById('inputText');
            inputElement.innerHTML = inputElement.textContent;
            inputElement.removeAttribute('contenteditable');
            const plainText = inputElement.innerHTML;

            const wordsToHighlight = value.filter(pair => pair[1] > threshold).map(pair => pair[0]);
            const escapeRegExp = (string) => string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const regex = new RegExp(`\\b(${wordsToHighlight.map(escapeRegExp).join('|')})\\b`, 'gi');        
            const highlightedText = plainText.replace(regex, '<span class="highlight">$&</span>');
            inputElement.innerHTML = highlightedText;

            toggleClickable();
        },
        function(error) {
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
    };

    const menuItemsContainer = document.getElementById('menu-items');
    callKNNFunction(event.target.textContent.trim()).then(
        function(items) {
            menuItemsContainer.innerHTML = items.map(item => `<li>${item}</li>`).join('');
            menuItemsContainer.querySelectorAll('li').forEach(li => {
                li.addEventListener('click', (e) => {
                    hideContextMenu();
                    event.target.textContent = e.target.textContent.trim();;
                    event.target.className = "";
                    event.target.removeEventListener('click', handleWordClick);
                });
            });
        },
        function(error) { console.error(error); }
    );

    document.addEventListener('click', (event) => {
        const insideMenu = contextMenu.contains(event.target);
        const onHighlight = event.target.classList.contains('highlight');
        if (!insideMenu && !onHighlight) { hideContextMenu(); }
    });

    contextMenu.style.display = 'block';
    contextMenu.style.left = `${event.pageX}px`;
    contextMenu.style.top = `${event.pageY}px`;
}

async function callTranslateFunction() {
    const apiUrl = 'http://localhost:8080/translate';
    const inputElement = document.getElementById('inputText');
    const outputElement = document.getElementById('outputText');
    const buttonElement = document.getElementById('buttonIcon');

    const className = buttonElement.className;
    buttonElement.className = 'fas fa-sync fa-spin';

    const requestData = {
        string: inputElement.textContent,
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
    // outputElement.textContent = JSON.stringify(data, null, 2);
    outputElement.textContent = json['output'];
    buttonElement.className = className;
    return json['scores'];
}

async function callKNNFunction(word) {
    const apiUrl = 'http://localhost:8080/neighbors';

    const requestData = {
        string: word,
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
    return json['neighbors'];
}
