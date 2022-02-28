let fs =  require("fs")
let network =  require("./nn2")

var dataFileBuffer  = fs.readFileSync(__dirname + '/train-images.idx3-ubyte');
var labelFileBuffer = fs.readFileSync(__dirname + '/train-labels.idx1-ubyte');
var pixelValues     = [];
var vdata =  []
// It would be nice with a checker instead of a hard coded 60000 limit here
for (var image = 0; image <= 200  ; image++) { 
    var pixels = [];

    for (var x = 0; x <= 27; x++) {
        for (var y = 0; y <= 27; y++) {
            pixels.push(dataFileBuffer[(image * 28 * 28) + (x + (y * 28)) + 15]/255);
        }
    }


    let n = parseInt(JSON.stringify(labelFileBuffer[image + 8]))
    if (image >= 950) {
        vdata.push([pixels,  Array(n).fill(0).concat([1]).concat(Array(9- n).fill(0)) ])

    }
    pixelValues.push([pixels, Array(n).fill(0).concat([1]).concat(Array(9- n).fill(0))] );
}
// let nt  =  new network([784,  30  , 10])


// nt.SGD(pixelValues , vdata, 400  , 30 ,  0.5  ,  5)
// fs.writeFileSync(__dirname+"/weights.json" , JSON.stringify(nt.weights))
// fs.writeFileSync(__dirname+"/biases.json" , JSON.stringify(nt.biases))



// let  g  = JSON.parse(fs.readFileSync(__dirname + '/weights.json'))
// nt.weights =  g ; 

// let  k  = JSON.parse(fs.readFileSync(__dirname + '/biases.json'))
// nt.biases = k  ; 
let  n  = 95
fs.writeFileSync(__dirname+"/d.js" , 'let  data  = '+JSON.stringify(pixelValues[n][0])+';module.exports = data ')
// let  data = pixelValues[n][0]
// let  out = nt.feedforward(data )

// let  h  =  0
// out.forEach(g => { 
//     if (g > h) h  = g
// })
// console.log(out)
// console.log(out.indexOf(h))





// let nt  =  new network([4,  2 , 1])
// data =  [
//     [[0 ,0 , 0 , 1] ,  [0]] , 
//     [[0 ,1 , 0 , 1] ,  [1]] ,
//     [[0 ,0 , 0 , 1] ,  [0]] ,
//     [[0 ,1 , 0 , 1] ,  [1]] ,
//     [[0 ,0 , 1 , 1] ,  [0]] ,
//     [[1 ,1 , 1 , 1] ,  [1]] 
// ]
// nt.SGD(data ,  700  ,  1 , 0.5)
// console.log(nt.feedforward([0 , 0 , 0 , 1]))
