let confidenceThreshold = 8.38; // 8.382382382382382
let frequencyThreshold = 1; // out-of-vocabulary

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
    callTranslateFunction().then(
        function(response) {
            const inputElement = document.getElementById('inputText');
            inputElement.innerHTML = inputElement.textContent;
            inputElement.removeAttribute('contenteditable');
            const plainText = inputElement.innerHTML;

            const wordsToHighlight = response['scores'].filter(pair => pair[1] > confidenceThreshold).map(pair => pair[0]);
            const escapeRegExp = (string) => string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            if (wordsToHighlight.length > 0) {
                const regex = new RegExp(`\\b(${wordsToHighlight.map(escapeRegExp).join('|')})\\b`, 'gi');        
                inputElement.innerHTML = plainText.replace(regex, match => {
                    let opacityStyle = '';
                    if (response['counts'][match] < frequencyThreshold) {
                        opacityStyle = 'opacity: 0.5;';
                    }
                    return `<span class="highlight" style="${opacityStyle}">${match}</span>`;
                });
            } else {
                inputElement.innerHTML = plainText;
            }

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
    event.target.style.cursor = 'wait';
    callNeighborsFunction(event.target.textContent.trim()).then(
        function(response) {
            event.target.style.cursor = 'default';

            menuItemsContainer.innerHTML = response['neighbors'].map(item => `<li>${item}</li>`).join('');
            menuItemsContainer.querySelectorAll('li').forEach(li => {
                li.addEventListener('click', (e) => {
                    hideContextMenu();
                    event.target.textContent = e.target.textContent.trim();;
                    event.target.className = '';
                    event.target.removeEventListener('click', handleWordClick);
                });
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
        function(error) { 
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
    return json;
}

async function callNeighborsFunction(word) {
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

// document.addEventListener('DOMContentLoaded', function () {
//     if (window.matchMedia('(orientation: landscape)').matches) {
//         document.querySelectorAll('details').forEach(detail => {
//             detail.setAttribute('open', '');
//         });
//     }
// });

function updateCollapsibleWidth() {
    const collapsible = document.querySelector('.collapsible');
    const container = document.querySelector('.container');
    if (collapsible && container) {
        collapsible.style.width = `${container.offsetWidth * 0.8}px`;
    }
}

window.addEventListener('resize', updateCollapsibleWidth);
window.addEventListener('load', updateCollapsibleWidth);

document.addEventListener('DOMContentLoaded', function () {
    const details = document.getElementById('info-details');
    const scrollIndicator = document.getElementById('scroll-indicator');

    function updateScrollIndicator() {
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

    // scrollIndicator.addEventListener('click', function () {
    //     details.scrollBy({ top: 100, behavior: 'smooth' });
    // });
    scrollIndicator.addEventListener('click', function () {
        details.scrollTo({ top: details.scrollHeight, behavior: 'smooth' });
    });    

    updateScrollIndicator();
});
