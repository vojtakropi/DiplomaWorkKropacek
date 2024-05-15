from utils import eval_results, test_model, load_model, load_weights, compile_model, eval_test_results
from models.UNet import UNet
from models.Unet_ribs import UNet_ribs
from models.xUNetFS import xUNetFS
from models.CNN import CNN
def main():
    # Define the name of the model
    model_name='ribs'
    
    #model = DeBoNet(COMPILE=False, NAME=model_name)        # If DeBoNet
    model = UNet_ribs()
    
    # load_weights(model, "model_tf_checkpoints/512/"+model_name+"/Unet_b10_f128_best_weights_28.weights.h5")

    ## INTERNAL TEST SET
    eval_test_results(model, model_name, RGB=False)


    
if __name__ == "__main__":
    main()
