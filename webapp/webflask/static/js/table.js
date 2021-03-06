$(document).ready(function () {
    $('#datatable').DataTable({
        data: record

        ,columns:[
            {title: "Title"},
            {title: "save_date"},
            {title: "path"},
            {title: "comment"}
        ]
        ,searching: false
        ,responsive: true
        ,"autoWidth": false

        ,dom: '<"float-left"B><"float-right"f>rt<"row"<"col-sm-4"l><"col-sm-4"i><"col-sm-4"p>>'
        ,language: {
                search: "_INPUT_", //To remove Search Label
                searchPlaceholder: "Search..."
        }

        ,fixedHeader: true

        ,fixedColumns:   {
            leftColumns: 1,
            rightColumns: 1
        }


    });

var table = $('#datatable').DataTable();
    $('#datatable tbody').on('click', 'tr', function () {
        $(".modal-body div span").text("");
        $(".title span").text(table.row(this).data()[0]);
        $(".saveDate span").text(table.row(this).data()[1]);
        $(".path span").text(table.row(this).data()[2]);
        $(".comment input").val(table.row(this).data()[3]);
        $(".a").attr("src", table.row(this).data()[2]+"/"+table.row(this).data()[0]);
        $(".b").attr("src", table.row(this).data()[2]+"/"+table.row(this).data()[4]);
        $(".c").attr("src", table.row(this).data()[2]+"/"+table.row(this).data()[5]);
        $(".hide input").val(table.row(this).data()[0]);
        $("#myModal").modal("show");
    });

    /* Column별 검색기능 추가 */
    $('#datatable').prepend('<select id="select"></select>');
    $('#datatable > thead > tr').children().each(function (indexInArray, valueOfElement) {
        $('#select').append('<option>'+valueOfElement.innerHTML+'</option>');
    });

});