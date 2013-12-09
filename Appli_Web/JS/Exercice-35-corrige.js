// Firefox 16.0.1

// Affiche la ListeDefinition n° i
function liste(i) {
	var definition = getDefinition(i)
	var contents = document.getElementById('contents')
	for (var i=contents.children.length-1 ; i >= 0 ; i--) {
		contents.removeChild(contents.children[i])
	}
	var child = createChild(definition)
	contents.appendChild(child)
}

// Crée récursivement les éléments HTML correspondants à la ListeDefinition 'definition'
function createChild(definition) {
	var dl = document.createElement('dl')
	var dt = document.createElement('dt')
	var dd = document.createElement('dd')
	dl.appendChild(dt)
	dl.appendChild(dd)
	dt.appendChild(document.createTextNode(definition.title))

	var ul = document.createElement('ul')
	dd.appendChild(ul)
	for (var i=0 ; i<definition.items.length ; i++) {
		var item = definition.items[i]
		var li = document.createElement('li')
		if (item.title) {
			var x = createChild(item)
			li.appendChild(x)
		} else {
			li.appendChild(document.createTextNode(item))
		}
		ul.appendChild(li)
	}

	return dl
}