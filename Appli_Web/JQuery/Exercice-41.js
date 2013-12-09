/* Cacher les continents vides */
function cacherVide() {
	$('tbody').each(function(i, elem){
		if ($(elem).children().size() == 0){
			$(elem).parents('div').hide();
		}
	}) 
}

/* Applique des styles differents (modulo 3) sur les lignes des tableaux */
function colorerLignes() {
	// TODO 2 
}

/* Aligne les colonnes 2 et 3 du <tbody> a droite, en appliquant la classe "nombre" aux <td> correspondants */
function nombres() {
	$('tr').children('td').eq(1).addClass('nombre');
	$('tr').children('td').eq(2).addClass('nombre');
}

/* Raccourcit le tableau passe en parametre (objet JQuery) en ne laissant que les 5 premieres lignes visibles. */
function raccourcir($tableau) {
	// TODO 4
}

/* Rallonge un tableau precedemment raccourci */
function rallonger($tableau) {
	// TODO 5
}

/* Initialise les clics sur les en-tetes des tableaux pour les rallonger / raccourcir */
function initClicEnteteTableaux() {

}

/* Gere le clic sur un <h2> */
function clicH2() {
	$(this).animate({'padding-left': '20px'}, 1000);
	$(this).animate({'padding-left': '0px'}, 1000);
}

/* Initialise les clics sur les <h2> */
function initClicH2() {
	$('h2').on('click', clicH2 );
}

/* Voici le code execute juste apres le chargement de la page : il n'y a rien a modifier ici */
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