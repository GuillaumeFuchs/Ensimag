// Firefox 16.0.1

// Affiche ou non le menu en fonction de l'état de la case à cocher
function switchMenu() {
	var coche = document.getElementById('showMenu')
	var menu = document.getElementById('menu')
	if (coche.checked) {
		menu.style.display = null
	} else {
		menu.style.display = "none"
	}
}

// Modifie le thème en jouant sur la classe de <body>
function switchTheme() {
	// Trouver le thème choisi
	var liste = document.getElementsByTagName('select')[0]
	var option
	for (i=0 ; i<liste.children.length ; i++) {
		if (liste.children[i].selected) {
			option = liste.children[i]
			break;
		}
	}

	// Modifier la classe de <body>
	var theme = option.value
	var body = document.getElementsByTagName('body')[0]
	body.className = theme

	// Cacher la case à cocher pour le menu dans theme1
	var coche = document.getElementById('showMenu')
	if (theme == "theme1") {
		coche.parentElement.style.display = "none"
	} else {
		coche.parentElement.style.display = null
	}

}

// Initialise l'affichage au chargement de la page
function initialize() {
	switchMenu()
	switchTheme()
}