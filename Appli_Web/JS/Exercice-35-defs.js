var list1 = { 'title' : 'Primary colors', 'items' : [ 'red', 'green', 'blue' ] }
var list2 = { 'title' : 'Detailed primary colors', 'items' : [ { 'title' : 'red', items: ['255','0','0']}, { 'title' : 'green', items: ['0','255','0']}, { 'title' : 'blue', items: ['0','0','255']} ] }
var list3 = { 'title' : 'Empty', 'items' :[] }
var list4 = { 'title' : '5MMCAW', 'items' : [ { 'title' : 'Course 1', items: [{ 'title' : 'Web applications', items: ['why','how']},{ 'title' : 'HTML', items: ['tags']}]} ] }
var list5 = { 'title' : '5MMCAW', 'items' : [ { 'title' : 'Course 1', items: [{ 'title' : 'Web applications', items: ['why','how']},{ 'title' : 'HTML', items: ['tags']}]}, 
			{ 'title' : 'Course 2', items: [{ 'title' : 'CSS', items: ['selectors','properties']},'Usability']}  ] }

function getDefinition(i) {
	switch (i) {
		case 1 : return list1
		case 2 : return list2
		case 3 : return list3
		case 4 : return list4
		case 5 : return list5
	}
}