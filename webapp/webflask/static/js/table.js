
var record = [[1, "abc", "data1", "image1", "comment1"],
    [2, "title2", "data2", "image2", "comment2"],
    [3, "title3", "data3", "image3", "comment3"],
    [4, "title4", "data4", "image4", "comment4"],
    [5, "title5", "data5", "image5", "comment5"],
    [6, "title6", "data6", "image6", "comment6"],
    [7, "title7", "data7", "image7", "comment7"],
    [8, "title8", "data8", "image8", "comment8"],
    [9, "title9", "data9", "image9", "comment9"],
    [10, "title10", "data10", "image10", "comment10"],
    [11, "title11", "data11", "image11", "comment11"],
    [12, "title12", "data12", "image12", "comment12"],
];

$(document).ready(function () {
    $('#datatable').DataTable({
        data: record,
        columns:[
            {title:"No"},
            {title: "Title"},
            {title: "Data"},
            {title: "Image"},
            {title: "Comment"}
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