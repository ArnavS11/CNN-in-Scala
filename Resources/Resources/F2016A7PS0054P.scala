package pplAssignment

object F2016A7PS0054P {

 /** Helper functions **/

  // Used to calculate dot product of lists
  def multiplyLists(a : List[Double], b : List[Double]) : Double =  a match {
  	case h::t => {
  		b match {
  			case head::tail => {
  				h*head + multiplyLists(t,tail);
  			}
  			case Nil => 0
  		}
  	}
  	case Nil => 0
  }

  // Used to add or multiply a constant to the list and return the resultant list
  def operationOnList(ls : List[Double], func : (Double) => Double) : List[Double] = ls match {
	case h::t => {
		val p = func(h);
		p :: operationOnList(t,func);
	}
    case Nil => Nil;
  }

  // Used to map the elements of a list (for min or max or sum) to a single value
  def mapList(ls : List[Double],func : (Double,Double) => Double) : Double = ls match {
	case h::t => {
		if(t == Nil)
	    	h;
		else {
		    val p = mapList(t,func);
			func(h,p);
		}
	}
	case Nil => -1; // To avoid warning in compilation
  }

  // Used to extract a sublist from a list
  def extractSublist(a : List[Double], currentElement : Int, reqElements: Int,displacement : Int) : List[Double] = a match {
  	case h :: t => {
  		if(displacement != 0)
  			return extractSublist(t,currentElement,reqElements,displacement-1);
  		if(currentElement == reqElements)
  			h :: Nil;
  		else {
  			val partialResult = extractSublist(t,currentElement+1,reqElements,displacement)
  			if(partialResult == Nil)
  				Nil;
  			h :: partialResult;
  		}
  	}
  	case Nil => Nil
  }

 /****/

 /** Convolution Layer **/

  def dotProduct(a : List[List[Double]], b : List[List[Double]]) : Double = a match {
  	case h::t => {
  		b match {
  			case head::tail => {
  				multiplyLists(h,head)+dotProduct(t,tail);
  			}
  			case Nil => 0
  		}
  	}
  	case Nil => 0
  }                                               //> dotProduct: (a: List[List[Double]], b: List[List[Double]])Double

  def extractMatrix(a : List[List[Double]],currentRow : Int,reqRows : Int, reqColumns : Int, displacement : Int, result : List[List[Double]]) : List[List[Double]] = a match {
  	case h::t => {
  		val extractedRow = extractSublist(h,1,reqColumns,displacement) :: Nil;
  		if(currentRow == reqRows)
  			extractedRow ::: result;
  		else {
  			val partialResult = extractMatrix(t,currentRow+1,reqRows,reqColumns,displacement,result);
  			extractedRow ::: partialResult;
  		}
  	}
  	case Nil => Nil
  }

  def getRow(a : List[List[Double]],kernel : List[List[Double]], kernelSize : List[Int], displacement_x : Int) : List[Double] = {
  	if(displacement_x < 0)
  		Nil;
  	else {
  		val matrix_1 = extractMatrix(a,1,kernelSize.head,kernelSize.tail.head,displacement_x,Nil);
  		val scalar = dotProduct(matrix_1,kernel) :: Nil;
  		getRow(a,kernel,kernelSize,displacement_x-1) ::: scalar;
  	}
  }

  def extract(a : List[List[Double]],kernel : List[List[Double]],column : Int,displacement_x: Int,displacement_y : Int,kernelSize : List[Int],result: List[List[Double]]) : List[List[Double]] = a match {
  	case h::t => {
  		if(column > displacement_y)
  			return Nil;
  		val resultRow = getRow(a,kernel,kernelSize,displacement_x);
  		resultRow :: extract(t,kernel,column+1,displacement_x,displacement_y,kernelSize,result);
  	}
  	case Nil => Nil;
  }

  def convolute(image : List[List[Double]], kernel : List[List[Double]], imageSize : List[Int], kernelSize : List[Int]) : List[List[Double]] =  {
  	val imageRows = imageSize.head; val imageColumns = imageSize.tail.head;
  	val kernelRows = kernelSize.head; val kernelColumns = kernelSize.tail.head;
  	val displacement_x = imageRows - kernelRows;
  	val displacement_y = imageColumns - kernelColumns+1;
  	extract(image,kernel,1,displacement_x,displacement_y,kernelSize,Nil);
  }

  /****************************************************/


  /**Activation Layer**/
  def applyActivation(ls : List[Double],activationFunc : Double => Double) : List[Double] = ls match {
  	case h :: t => {
  		activationFunc(h) :: applyActivation(t,activationFunc);
  	}
  	case Nil => Nil
  }

  def activationLayer(activationFunc : Double => Double,image: List[List[Double]]) : List[List[Double]] =  {
  	image match {
	  	case h::t => {
	  		val ls = applyActivation(h,activationFunc);
	  		ls :: activationLayer(activationFunc,t);
	  	}
	  	case Nil => Nil;
  	}
  }

  /*************************/


  /*********Pooling Layer**********/
  def getList(image: List[List[Double]],current_row : Int, current : Int, K : Int) : List[Double] =  image match {
  	case h::t => {
  		if(current_row == K)
  			Nil;
  		else {
	  		val row = extractSublist(h,1,K,current);
	  		if(row == Nil)
	  			Nil;
	  		row ::: getList(t,current_row+1,current,K);
  		}
  	}
  	case Nil => Nil;
  }

  def extractForSinglePooling(poolingFunc : List[Double] => Double, image : List[List[Double]], current : Int, K : Int) : List[Double] = {
  	val ls = getList(image,0,current,K);
  	if(ls == Nil)
  		Nil;
  	else {
  		poolingFunc(ls) :: extractForSinglePooling(poolingFunc,image,current+K,K);
  	}
  }

  def singlePooling(poolingFunc : List[Double] => Double, image : List[List[Double]], K : Int) : List[Double] = {
  	extractForSinglePooling(poolingFunc,image,0,K);
  }

  def extractForPoolingLayer(poolingFunc : List[Double] => Double,image : List[List[Double]], currentRow : Int, K : Int) : List[List[Double]] = image match {
  	case h::t => {
  		if(currentRow%K == 0) {
  			val singlePool = singlePooling(poolingFunc,image,K);
  			val ls = extractForPoolingLayer(poolingFunc,t,currentRow+1,K);
  			singlePool :: ls;
  		}
  		else {
  			extractForPoolingLayer(poolingFunc,t,currentRow+1,K);
  		}
  	}
  	case Nil => Nil;
  }

  def poolingLayer(poolingFunc : List[Double] => Double, image : List[List[Double]], K : Int) : List[List[Double]] = {
  	extractForPoolingLayer(poolingFunc,image,0,K);
  }

  /*****************************/


  /****MixedLayer****/
  def mixedLayer(image : List[List[Double]], kernel : List[List[Double]], imageSize : List[Int], kernelSize: List[Int],activationFunc : (Double) => Double, poolingFunc : (List[Double]) => Double, K : Int) : List[List[Double]] = {
  	val convoluted : List[List[Double]] = convolute(image,kernel,imageSize,kernelSize);
  	val activated = activationLayer(activationFunc,convoluted);
  	poolingLayer(poolingFunc,activated,K)
  }

  /*****************/

  /***Normalisation***/

  def getRange(image: List[List[Double]]) : List[Double] = image match {
  	case h :: t => {
  		val max = mapList(h,(x:Double,y:Double) => if(x>y) x else y);
  		val min = mapList(h,(x:Double,y:Double) => if(x<y) x else y);

  		if(t == Nil)
  			min :: max :: Nil;
  		else {
  			val ls = getRange(t);
  			if(ls == Nil)
  				(min :: max :: Nil);
  			else {
  				if(ls.head <= min && ls.tail.head >= max)
  					ls.head :: ls.tail.head :: Nil;
  				else if(ls.head <= min && max >= ls.tail.head)
  					ls.head :: max :: Nil;
  				else if(ls.head >= min && max <= ls.tail.head)
  					min :: ls.tail.head :: Nil;
  				else
    					min :: max :: Nil;
  			}
  		}
  	}
  	case Nil => Nil;
  }

  def normaliseList(ls : List[Double], matMin : Double, matMax : Double, rangeMin : Int, rangeMax : Int) : List[Int] = ls match {
  	case h :: t => {
  		val num  = Math.round(((h - matMin)*(rangeMax - rangeMin))/(matMax - matMin) + rangeMin);
  		num.toInt :: normaliseList(t,matMin,matMax,rangeMin,rangeMax);
  	}
  	case Nil => Nil;
  }

  def normaliseMatrix(image: List[List[Double]], matMin : Double, matMax : Double, rangeMin : Int, rangeMax : Int ) : List[List[Int]] = image match {
  	case h::t => {
  		val ls = normaliseList(h,matMin,matMax,rangeMin,rangeMax);
  		ls :: normaliseMatrix(t,matMin,matMax,rangeMin,rangeMax);
  	}
  	case Nil => Nil;
  }

  def normalise(image : List[List[Double]]) : List[List[Int]] = {
  	val range = getRange(image);
		val min = range.head; val max = range.tail.head;
		normaliseMatrix(image,min,max,0,255);
  }
  /*********/


  /*****Assembly*****/

  def multiplyWeight(image : List[List[Double]], weight : Double) : List[List[Double]] = image match {
  	case h::t => {
  		val ls = operationOnList(h,(x : Double) => x*weight);
  		ls :: multiplyWeight(t,weight);
  	}
  	case Nil => Nil;
  }

  def addBias(image : List[List[Double]], b : Double) : List[List[Double]] = image match {
  	case h::t => {
  		val p = operationOnList(h,(x : Double) => x + b);
  		p :: addBias(t,b);
  	}
  	case Nil => Nil;
  }

  def addLists(ls1 : List[Double], ls2 : List[Double]) : List[Double] = ls1 match {
  	case h::t => {
  		ls2 match {
  			case head::tail => (h+head) :: addLists(t,tail);
  			case Nil => h :: addLists(t,Nil);
  		}
  	}
  	case Nil => {
  		ls2 match {
  			case head :: tail => head ::addLists(Nil,tail);
  			case Nil => Nil;
  		}
  	}
  }

  def addMatrices(mat1 : List[List[Double]], mat2 : List[List[Double]]) : List[List[Double]] = mat1 match{
  	case h::t => {
  		mat2 match {
  			case head :: tail => {
  				val pls = addLists(h,head);
  				pls :: addMatrices(t,tail);
  			}
  			case Nil => h :: addMatrices(t,Nil);
  		}
  	}
  	case Nil => {
  		mat2 match {
  			case head::tail => head::addMatrices(Nil,tail);
  			case Nil => Nil;
  		}
  	};
  }


  def assembly(image: List[List[Double]], imageSize : List[Int], w1 : Double, w2 : Double, b : Double, kernel1 : List[List[Double]], kernelSize1 : List[Int], kernel2 : List[List[Double]], kernelSize2 : List[Int], kernel3 : List[List[Double]], kernelSize3 : List[Int], size : Int) : List[List[Int]] = {
  	val temp_output1 = mixedLayer(image,kernel1,imageSize,kernelSize1,relu,avg_pooling,size);
  	val temp_output2 = mixedLayer(image,kernel2,imageSize,kernelSize2,relu,avg_pooling,size);
  	val input_mat = addBias(addMatrices(multiplyWeight(temp_output1,w1),multiplyWeight(temp_output2,w2)),b);
  	val rows = (input_mat).length; val columns = (input_mat.head).length;
  	val temp_output3 = mixedLayer(input_mat,kernel3,rows::columns::Nil,kernelSize3,leaky_relu,max_pooling,size);
  	normalise(temp_output3);
  }

  /******/

  /**Activation and Pooling functions**/
  	def relu(x : Double) : Double = {
  		if(x < 0)
  			0;
		else
  			x;
  	}                                         //> relu: (x: Double)Double

  	def leaky_relu(x : Double) : Double = {
  		if(x >= 0)
  			x;
		else
  			0.5*x;
  	}                                         //> leaky_relu: (x: Double)Double

  	def max_pooling(ls : List[Double]) : Double = {
			mapList(ls,(x:Double,y:Double)=> if(x>y) x else y);
  	}                                         //> max_pooling: (ls: List[Double])Double

  	def avg_pooling(ls : List[Double]) : Double = {
			val len = ls.length;
			val sum = mapList(ls,(x : Double,y : Double) => x+y);
			sum/len;
  	}                                         //> avg_pooling: (ls: List[Double])Double
  /*****/
}
