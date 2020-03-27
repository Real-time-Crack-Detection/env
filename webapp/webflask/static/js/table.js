
$(document).ready(function () {
    $('#datatable').DataTable({
        data: record,
        columns:[
            {title:"Title"},
            {title: "save_date"},
            {title: "path"},
            {title: "comment"}
        ]
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

    /* Column별 검색기능 추가 */
    $('#datatable').prepend('<select id="select"></select>');
    $('#datatable > thead > tr').children().each(function (indexInArray, valueOfElement) {
        $('#select').append('<option>'+valueOfElement.innerHTML+'</option>');
    });

});