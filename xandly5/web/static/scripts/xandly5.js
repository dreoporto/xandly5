function submitForm(){

    $("#generated_text").text("Processing request...")

    lyricsRequest = {
        model_id: parseInt($("#model_id").val()),
        seed_text: $("#seed_text").val(),
        word_count: parseInt($("#word_count").val()),
        word_group_count: parseInt($("#word_group_count").val())
    };

    //TODO AEO TEMP
    //debugger;
    console.log(JSON.stringify(lyricsRequest));

    $.ajax({
        url: '/lyrics-api',
        data: JSON.stringify(lyricsRequest),
        contentType: 'application/json',
        type: 'POST',
        success: function(xhr){ $("#generated_text").text(xhr); }
    });
}

//TODO AEO TEMP - basic code for now; no validation or wait spinner
$("#main_button").click(function(){
    submitForm();
});