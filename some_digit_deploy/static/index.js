(function(){
    var canvas = document.querySelector('#digito_canvas');
    var context = canvas.getContext('2d');

    canvas.width = 280;
    canvas.height = 280;

    var Mouse = { x : 0, y : 0};
    var lastMouse = { x : 0, y : 0};

	context.fillStyle="white";
	context.fillRect(0,0,canvas.width,canvas.height);
	context.color = "black";
	context.lineWidth = 10;
    context.lineJoin = context.lineCap = 'round';

    debug();

    canvas.addEventListener('mousemove', function(e){
        lastMouse.x = Mouse.x;
        lastMouse.y = Mouse.y;

        Mouse.x = e.pageX - this.offsetLeft;
        Mouse.y = e.pageY - this.offsetTop; 

    }, false);

    canvas.addEventListener('mousedown', function(){
        canvas.addEventListener('mousemove', onPaint, false);
    }), false;

    canvas.addEventListener('mouseup', function(){
        canvas.removeEventListener('mousemove', onPaint, false);
    }), false;


    var onPaint = function(){
        context.lineWidth = context.lineWidth;
        context.lineJoin = "round";
		context.lineCap = "round";
		context.strokeStyle = context.color;

        context.beginPath();
        context.moveTo(lastMouse.x, lastMouse.y);
        context.lineTo(Mouse.x, Mouse.y);
        context.stroke();
        context.closePath();
    }

    function debug()
	{
		/* CLEAR BUTTON */
		var clearButton = $( "#clearButton" );
		
		clearButton.on( "click", function()
		{
				context.clearRect( 0, 0, 280, 280 );
				context.fillStyle="white";
				context.fillRect(0,0,canvas.width,canvas.height);
		});

		$( "#colors" ).change(function()
		{
			var color = $( "#colors" ).val();
			context.color = color;
		});
		
		$( "#lineWidth" ).change(function()
		{
			context.lineWidth = $( this ).val();
		});
	}
})();