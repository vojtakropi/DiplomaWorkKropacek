from utils import check_capabilities, load_model, load_weights, compile_model, train_model, save_model, save_weights, train_debonet, train_kalisz

from models.xUNetFS import xUNetFS

def main():
    ## Check the availability of GPU
    check_capabilities()
    # strategy = tf.distribute.MirroredStrategy()
    # print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    # with strategy.scope():
    ## Define the model
    model = xUNetFS()
    
    ## Load weights if needed
    #load_weights(model, "PATH_TO_WEIGHTS")

    # Compile
    model = compile_model(model)

    
    # Training
    #history = train_debonet(model)     ## For DeBoNet
    history = train_model(model)
    #history = train_kalisz(model)      ## For Kalisz Marczyk Autoencoder
    save_model(history, "unetFS")
    save_weights(history, "unetweights")
if __name__ == "__main__":
    main()
