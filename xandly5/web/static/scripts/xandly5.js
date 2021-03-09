function submitForm(){

    $("#generated_text").text("Processing...")
    $("#main_button").prop("disabled", true);
    $('#spinner').removeClass("d-none");
    $('#alert_message').hide();

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
        success: function(xhr){
            $('#spinner').addClass("d-none");
            $("#generated_text").text(xhr);
            $("#main_button").prop("disabled", false);
        },
        error: function(xhr){
            $("#generated_text").text('');
            $('#alert_message').text(xhr.responseJSON.message);
            $('#alert_message').show();
            $('#spinner').addClass("d-none");
            $("#main_button").prop("disabled", false);
        }
    });
}

//TODO AEO TEMP - add UI validations
$("#main_button").click(function(){
    submitForm();
});