/* Cacher les continents vides */
function cacherVide() {
	// Trouver tous les <tbody>, les parcourir, et pour ceux sans enfants, cacher le <div> ascendant
	$('tbody').each(function(index, element) {
		if ($(element).children().size() == 0) {
			$(element).parents('div.continent').hide()
		}
	})
}

/* Applique des styles differents (modulo 3) sur les lignes des tableaux */
function colorerLignes() {
	// Parcourir les tableaux
	$('tbody').each(function(i, tbody) {
		// Pour chaque tableau, parcourir les lignes et leur appliquer la classe ligX
		// ou X vaut 1,2,3 puis a nouveau 1,2,3 etc.
		var num = 1
		$(tbody).find('tr').each(function(j,tr) {
			// Supprimer toute classe ligX de la ligne
			$(tr).removeClass('lig1')
			$(tr).removeClass('lig2')
			$(tr).removeClass('lig3')
			
			// Ajouter la classe correspondante a la ligne courante
			$(tr).addClass('lig' + num++)
			if (num == 4) num = 1
		})
	}) 
}

/* Aligne les colonnes 2 et 3 du <tbody> a droite, en appliquant la classe "nombre" aux <td> correspondants */
function nombres() {
	$('tbody td + td').addClass('nombre') 
}

/* Raccourcit le tableau passe en parametre (objet JQuery) en ne laissant que les 5 premieres lignes visibles. */
function raccourcir($tableau) {
	// Cacher les lignes au-dela de la cinquieme
	var cache = false
	$tableau.find('tbody tr').each(function(i,tr) {
		if (i > 4) {
			$(tr).hide()
			cache = true
		}
	})

	// Afficher le fait que le tableau a ete raccourci
	if (cache) {
		$tableau.find('thead').after('<tfoot><tr><th>...</th><th>...</th><th>...</th></tr></tfoot>')
		$tableau.addClass('raccourci')
	}
}

/* Rallonge un tableau precedemment raccourci */
function rallonger($tableau) {
	$tableau.removeClass('raccourci')
	$tableau.find('tfoot').remove()
	$tableau.find('tbody tr').show()
}

/* Initialise les clics sur les en-tetes des tableaux pour les rallonger / raccourcir */
function initClicEnteteTableaux() {
	$('table thead th').on('click', function() { 
		$table = $(this).parents('table')
		if ($table.hasClass('raccourci'))
			rallonger($table)
		else
			raccourcir($table)
	})
	$('table').addClass('cliquable')
}

/* Gere le clic sur un <h2> */
function clicH2() {
	$h2 = $(this)
	$h2.animate(
		{ paddingLeft : "10em" }, 
		1000, 
		function() { 
			$h2.animate( 
				{ paddingLeft : 0 }, 
				1000)
		}
	)
}

/* Initialise les clics sur les <h2> */
function initClicH2() {
	$('h2').on('click', clicH2)
}

/* Voici le code execute juste apres le chargement de la page */
$(function() {
	// cacher les sections vides
	cacherVide()

	// colorer les lignes des tableaux
	colorerLignes()

	// aligner les nombres a droite
	nombres()

	// raccourcir tous les tableaux
	$('table').each(function(i,t) { raccourcir($(t)) })

	// initialiser la possibilite de cliquer sur les en-tetes de tableaux pour les rallonger/raccourcir
	initClicEnteteTableaux()

	// initialiser la possibilite de cliquer sur les titres <h2>
	initClicH2()
})