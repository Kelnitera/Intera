import tensorflow as tf
import json, requests

def Modeler(test_data, url, output="max"):
  data = json.dumps({"signature_name":"serving_default", "instances":test_data})
  json_resp = requests.post(url, data=data, headers={"content-type":"application/json"})
  json_load = json.loads(json_resp.text)
  result = json_load["predictions"]
  if output == "max":
    return tf.math.argmax(result, axis=1).numpy()
  elif output == "original":
    return result