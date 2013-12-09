// Firefox 16.0.1

function lifo_push() {
	var text_input = document.getElementById('newItem')
	var lifo = document.getElementById('lifo')
	var li = document.createElement('li')
	li.appendChild(document.createTextNode(text_input.value))
	if (lifo.childElementCount == 0) 
		lifo.appendChild(li)
	else {
		var first = lifo.firstElementChild
		lifo.insertBefore(li, first)
	}
	text_input.value = ''
}

function lifo_pop() {
	var lifo = document.getElementById('lifo')
	if (lifo.childElementCount == 0) 
		alert('La pile est vide !')
	else
		lifo.removeChild(lifo.firstElementChild)
}

function lifo_peek() {
	var pa = document.getElementById('peek_area')
	var lifo = document.getElementById('lifo')
	if (lifo.childElementCount == 0) 
		alert('La pile est vide !')
	else {
		var first = lifo.firstElementChild
		if (pa.firstChild)
			pa.removeChild(pa.firstChild)
		pa.appendChild(document.createTextNode(first.innerHTML))
	}
}