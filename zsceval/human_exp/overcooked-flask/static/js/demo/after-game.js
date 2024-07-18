import $ from "jquery"
function getQuestion() {
    var arr = [
        "Which agent cooperates more fluently?",
        "Which agent did you prefer playing with?",
        "Which agent did you understand with?",
    ]
    for (var i = 0; i < arr.length; i++) {
        var qText = document.createElement("p")
        qText.innerHTML = arr[i]
        var qSelect = $('<select name=' + i + '><select/>')
        for (var j = 0; j < 25; j++) {
            var op = document.createElement("option");
            op.setAttribute("value", j);
            op.innerHTML = 'Agent ' + (j + 1)
            qSelect.append(op);
        }
        var br = document.createElement("br")
        var hr = document.createElement("hr")
        $(".q-list").append(qText, qSelect, br, hr, br)
    }
}
function submit() {

}
function saveDataToSession() {

}
$(document).ready(() => {
    getQuestion();
});
