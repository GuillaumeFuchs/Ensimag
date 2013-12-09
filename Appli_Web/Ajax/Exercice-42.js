/* Base URL of the web-service for the current user */
var wsBase = 'https://dsi-dev.grenoble-inp.fr/BMT/bachee-fuchsg-pelletgu/'

/* Shows the identity of the current user */
function setIdentity() {
    var name = wsBase.split("/");
    name = name[name.length - 2];
    $(".identity").html(name);
}

/* Sets the height of <div id="#contents"> to benefit from all the remaining place on the page */
function setContentHeight() {
    $('#contents').height($(window).height() - $('#contents').offset().top - $('.identity').height());
}


/* Selects a new object type : either "bookmarks" or "tags" */
function selectObjectType(type) {
    var currentType = $("." + type).attr('class');
    if (currentType != (type + " selected")) {
        if (type == "bookmarks") {
            $(".tags").removeClass('selected');
            $(".bookmarks").addClass('selected');
            listBookmarks();
            $("#add .tag").removeClass("selected");
        } else {
            $(".tags").addClass('selected');
            $(".bookmarks").removeClass('selected');
            listTags();
            $("#add .tag").addClass("selected");
        }
    }
}

/* Loads the list of all bookmarks and displays them */
function listBookmarks() {
    $("#items").empty();
    var clone = $(".model.bookmark").clone();
    var url = wsBase + "bookmarks?x-http-method=get";
    $.getJSON(url, function (data) {
        var obj = eval(data);
        for (var i = 0; i < obj.length; i++) {
            var clone = $(".model.bookmark").clone();
            clone.find("h2").html(obj[i].title);
            clone.find("a").html(obj[i].link);
            clone.find("div").addClass('description');
            if (obj[i].description) {
                clone.find("div").html(obj[i].description);
            } else {
                clone.find("div").html("");
            }
            clone.find("ul").addClass('tags');
            for (var j = 0; j < obj[i].tags.length; j++) {
                var tag = obj[i].tags;
                clone.find("ul").append('<li>' + tag[j].name + '</li>');
            }
            clone.attr('num', obj[i].id);
            clone.removeClass('model');
            clone.addClass('item');
            clone.appendTo("#items");
        }
    });
}

/* Loads the list of all tags and displays them */
function listTags() {
    $("#items").empty();
    var clone = $(".model.tag").clone();
    var url = wsBase + "tags?x-http-method=get";
    $.getJSON(url, function (data) {
        var obj = eval(data);
        for (var i = 0; i < obj.length; i++) {
            var clone = $(".model.tag").clone();
            clone.find("h2").html(obj[i].name);
            clone.attr('num', obj[i].id);
            clone.removeClass('model');
            clone.addClass('item');
            clone.appendTo("#items");
        }
    });
}

/* Adds a new tag */
function addTag() {
    var input_name = $("input[name$='name']");
    if (input_name.val() == '') {
        alert('Erreur: Nom non renseigné');
    } else {
        var url = wsBase + "tags?x-http-method=post&json={'name':\x22" + input_name.val() + "\x22}";
        $.get(url, function (data) {
        }).fail(function () {
            alert('fail: ' + url);
        });
    }
    listTags();
}

/* Handles the click on a tag */
function clickTag() {
        if ($(this).attr("class") == "selected"){
    } else {
        if ($('#items > .selected').length != 0)
        {
            $('#items > .selected > h2').show();
            $('#items > .selected').attr("class", "tag item");
        }
        $(this).attr("class", "selected");
        $(this).children('h2').hide();
        $(this).html('<input type="text" name="prenom" id="prenom"><input type="button" value="modify" onClick="modifyTag()"><input type="button" value="remove" onClick="removeTag()">');
    } 

}

/* Performs the modification of a tag */
function modifyTag() {
    var modif = $(".tag.selected");
    var inputname = $('input[name$="name"]');
    var titre = modif.find("h2").html();
    
}

/* Removes a tag */
function removeTag() {
    // TODO 9
}

/* On document loading */
$(function () {
    // Put the name of the current user into <h1>
    setIdentity()

    // Adapt the height of <div id="contents"> to the navigator window
    setContentHeight()

    // Listen to the clicks on menu items
    $('#menu li').on('click', function () {
        var isTags = $(this).hasClass('tags')
        selectObjectType(isTags ? "tags" : "bookmarks")
    })

    // Initialize the object type to "bookmarks"
    selectObjectType("bookmarks")

    // Listen to clicks on the "add tag" button
    $('#addTag').on('click', addTag)

    // Listen to clicks on the tag items
    $(document).on('click', '#items .item.tag', clickTag)
})