    # import tensorflow library
    import  tensorflow  as tf


    # Create a Constant op
    # The op is added as a node to the default graph.
    # The value returned by the constructor represents the output of the Constant op.
    hello_world = tf.constant('Hello, World!')
    
    # Start tf session
    sess = tf.Session()
    
    
    # Run graph# Run g 
    print(sess.run(hello_world))

Hello, World!