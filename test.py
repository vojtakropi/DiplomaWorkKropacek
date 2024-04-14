from utils import eval_results, test_model, load_model, load_weights, compile_model, eval_test_results, eval_test_results_woPred
from models.UNet import UNet
from models.xUNetFS import xUNetFS
def main():
    # Define the name of the model
    model_name='Unet'
    
    #model = DeBoNet(COMPILE=False, NAME=model_name)        # If DeBoNet
    model = UNet()
    
    load_weights(model, "model_tf_checkpoints/512/"+model_name+"/Unet_b10_f128_best_weights_99.weights.h5")

    ## INTERNAL TEST SET
    eval_test_results(model, model_name, RGB=False)         # RGB = True is necessary for DeBoNet models
    #eval_test_results_woPred("predictions/512/DEBONET/", "ribs_suppresion/new/augmented/test/BSE_JSRT/", model_name)       # If DeBoNet
    
    ## TESTING UNSEEN IMAGES -- EXTERNAL TEST SET
    result, ids = test_model(model)
    eval_results(result, ids, model_name)
    
if __name__ == "__main__":
    main()
