
function  shuffle(arr){
    let rarr =  [] , indexs =  []
    for (let i  =  0  ;  i<  arr.length ; i ++){
        let  index =  0 ; 
        do{     
            index  =  Math.floor(Math.random() *  arr.length)
        }while(indexs.indexOf(index) > - 1)
        indexs.push(index)
        rarr.push(arr[index])
    }
    return rarr
}



function  matrix (y  ,x ){
    let mtx =  []
    for  (let  i  = 0 ;  i <  y  ; i ++){
        let  ys = []
        for(let  j  =  0 ; j < x ; j++ ){
            ys.push(Math.random() * 2 * Math.pow(-1 , j))
        }
        mtx.push(ys)
    }
    return  mtx ; 
}

function  vmatrix (y  ,x , v){
    let mtx =  []
    for  (let  i  = 0 ;  i <  y  ; i ++){
        let  ys = []
        for(let  j  =  0 ; j < x ; j++ ){
            ys.push(v)
        }
        mtx.push(ys)
    }
    return  mtx ; 
}

function  nmatrix (y  ,x ){
    let mtx =  []
    for  (let  i  = 0 ;  i <  y  ; i ++){
        let  ys = []
        for(let  j  =  0 ; j < x[i + 1] ; j++ ){
            ys.push(Math.random() * 2)
        }
        mtx.push(ys)
    }
    return  mtx ; 
}
function  vnmatrix (y  ,x , v ){
    let mtx =  []
    for  (let  i  = 0 ;  i <  y  ; i ++){
        let  ys = []
        for(let  j  =  0 ; j < x[i + 1] ; j++ ){
            ys.push(v)
        }
        mtx.push(ys)
    }
    return  mtx ; 
}

//exec  a  fuction btwn two  matrixs
function  ematrix( m1  ,  m2  ,  fn){ //the two  matrix  must  be the same size
    for  (let  i  = 0 ;  i <  m1.length ; i ++){
        for(let  j  =  0 ; j < m1[i].length ; j++ ){
            m1[i][j] = fn(m1[i][j]  , m2[i][j] )
        }
    }
    return  m1 ; 
}




class network {

    constructor (sizes) {

        this.size  =  sizes.length 
        this.sizes =  sizes 
        this.biases =  nmatrix(this.size - 1 ,  sizes) //biases  start from 2nd  layer
        this.weights =  []
        for (let  l  =  0 ;  l <  this.size - 1 ; l ++){
            this.weights.push(matrix(sizes[l + 1] , sizes[l] ))
        }
    }

    evaluat (data) {
        let res =  0 ;
        for (let i  = 0 ; i< data.length ;  i++){
            let out = this.feedforward(data[i][0]) 
            if (out[data[i][1].indexOf(1)].toFixed(0)==1){
                res ++
            }
        }
        return  res ;
    }

    SGD (tr_data  , epochs  , mini_batch  , eta , vdata ){

        for  (let  e  =  0 ; e  < epochs; e ++ ){ //will train  our model for  epochs time hhhhh 
            let mini_batchs  =  []
            console.log("epoche : "+ e+"=========")
            
           // console.log(this.feedforward(tr_data[0][0]) ,  tr_data[0][1])
        //console.log(this.evaluat(vdata)+"/"+vdata.length , "rate = " + (this.evaluat(vdata) / vdata.length))
            //tr_data =  shuffle(tr_data)
            for (let  i  =  0  ;  i<tr_data.length ;  i  += mini_batch){
                mini_batchs.push(tr_data.slice(i  , i  + mini_batch))
            }
            
            for  ( let  j  =  0 ;  j  < mini_batchs.length ; j++ ){

                this.update_mini_batch( mini_batchs [j] ,  eta);
            }

        }

    }

            
    update_mini_batch  ( mini_batch ,  eta  ){
        // for each  mini  bacth  will  calculate  the  derivative  of the cost fn
        let  nabla_b =  vnmatrix(this.size - 1 ,  this.sizes , 0) //biases  start from 2nd  layer
        let  nabla_w =  []
        for (let  l  =  0 ;  l <  this.size - 1 ; l ++){
            nabla_w.push(vmatrix(this.sizes[l + 1] , this.sizes[l]  , 0))
        }
        //calculate the derivative for one  input
        for  (let  t  = 0 ; t <  mini_batch.length ; t  ++ ){
            let  delta_w  ,  delta_b 
            [delta_w , delta_b] = this.backprop( mini_batch[t][0] ,   mini_batch[t][1])
            nabla_b = ematrix(nabla_b , delta_b , (a  , b)=>  a+b)
            for (let  l  = 0 ;  l  < nabla_w.length ;  l ++ ){
                nabla_w[l] = ematrix(nabla_w[l] , delta_w[l] , (a ,b)=> a +b) 
            }
            
        }
        this.biases = ematrix(this.biases , nabla_b ,  
            (a ,b) => a  - ((eta /mini_batch.length) * b)) 
        
        for (let  l  = 0 ;  l  < nabla_w.length ;  l ++ ){
            nabla_w[l] = ematrix( this.weights[l], nabla_w[l] ,
                (a ,b) => a  - (eta * (b /mini_batch.length))) 
        }

    } 

    backprop  (x  , y ){
        let  nabla_xb =  vnmatrix(this.size - 1 ,  this.sizes , 0) //biases  start from 2nd  layer
        let  nabla_xw =  []
        for (let  l  =  0 ;  l <  this.size - 1 ; l ++){
            nabla_xw.push(vmatrix(this.sizes[l + 1] , this.sizes[l]  , 0))
        }


        //feedforward  
        let activetion  =  x , activetions =  [x] , zs = []
        for (let l  =  0 ;  l <  this.biases.length ;  l++ ){ //the  nth  layer  here is  actually (n+1)th in  the full network
            let activetion_l = [] // store  activetion  of  this  layer
            let z_layer = []
            for  (let  n  =  0  ;  n < this.biases[l].length ; n ++){
                let  somme =  0  
                for (let  wn = 0 ; wn <  this.weights[l][n].length ; wn ++){
                    somme  += ( activetion[wn] * this.weights[l][n][wn])
                }
                let  z  = somme  + this.biases[l][n] ;
                activetion_l.push(segmoid(z)) 
                z_layer.push(z)
            }
            zs.push(z_layer)
            activetion = activetion_l
            activetions.push(activetion_l)
        } 
        //back  pass 
        
        //delta for  the  output layer
        
        let delta  =  this.cost_derivative(activetions[activetions.length -1] , y)
        
        nabla_xb[nabla_xb.length - 1] = delta
        
       for  (let j =  0 ; j < delta.length ; j ++){
           for (let k =  0 ; k < nabla_xw[nabla_xw.length - 1][j].length ; k ++){
                nabla_xw[nabla_xw.length - 1][j][k] = activetions[activetions.length -2][k] * delta[j]  
           }
       }
       

       //now for  each  layer
       for  (let l =  2 ; l < this.size  ; l ++){
            let delta_l = []
            for  (let j =  0 ; j < zs[zs.length -l].length ; j ++){   
                let somme = 0
                for  (let k =  0 ; k < delta.length ; k ++){
                    somme  += delta[k] * this.weights[this.weights.length -l + 1][k][j]
                }
                delta_l.push( nsigmoid_prime(zs[zs.length -l][j]) * somme )
            }
            nabla_xb[nabla_xb.length - l] = delta_l
        
            for  (let j =  0 ; j < delta_l.length ; j ++){
                for (let k =  0 ; k < nabla_xw[nabla_xw.length - l ][j].length ; k ++){
                     nabla_xw[nabla_xw.length - l][j][k] = activetions[activetions.length -l -1][k] * delta_l[j]  
                }
            }

            delta = delta_l
        }

        return  [nabla_xw , nabla_xb]
    }


    cost_derivative(output_activations, y){
        let res =  [] 

        for (let  j = 0  ; j < y.length ;  j ++){
            res.push(output_activations[j] -  y[j])
        }
        return res 
    } 

    feedforward (input) {

        let activetion  =  input  
 
        for (let l  =  0 ;  l <  this.biases.length ;  l++ ){ //the  nth  layer  here is  actually (n+1)th in  the full network
            let activetion_l = [] // store  activetion  of  this  layer
            for  (let  n  =  0  ;  n < this.biases[l].length ; n ++){
                let  somme =  0  
                for (let  wn = 0 ; wn <  this.weights[l][n].length ; wn ++){
                    somme  += ( activetion[wn] * this.weights[l][n][wn])
                }
                activetion_l.push(segmoid(somme  + this.biases[l][n] )) 
            }
            activetion = activetion_l
        } 
        return  activetion ; 
    }
}

function  segmoid (z){
    return 1.0/(1.0+ Math.exp(-z))
}
function sigmoid_prime(z){
    let  res = []
    for  (let  j  = 0  ;  j <  z.length ; j++  ){
        res.push( segmoid(z[j])*(1-segmoid(z[j])))
    }
    return res

}
function nsigmoid_prime(z){
    return  segmoid(z)*(1-segmoid(z))

}
function  prvector(v1  ,  v2){ //two  vectors must  be  the  same size 
    let  res =  []
    for  (let  j  = 0  ;  j <  v1.length ; j++  ){
        res.push(v1[j] * v2[j])
    }
    return res
}

let  n =  new network([3 , 3 ,  1]) 

data =  [
    [[1  ,  0 ,  1, 1] , [0]] , 
    [[1  ,  1 ,  1, 1] , [1]] ,

]
n.SGD(data , 500 ,  2, 0.2)
console.log( n.feedforward([1 , 0 , 1 , 1]) )
module.exports =  network
