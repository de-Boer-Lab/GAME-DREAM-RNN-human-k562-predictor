'''
Decode Request and Encode Response in Negotiated MIME Type.
Always JSON for DREAM-RNN
'''

from flask import request, jsonify

from error_checking_functions import BadRequestError, ServerError

def decode_request(supported_request_formats):
    """
    Decodes the incoming request body based on Content-Type.
    Supports JSON only.

    Args:
        supported_request_formats (list): Lowercase list of supported request mime types

    Returns:
        dict: The decoded Python dictionary from the request body

    Raises:
        BadRequestError: If Content-Type is missing, unsupported, or decoding fails
    """
    content_type_header = request.headers.get('Content-Type')
    # If no header is present, try to decode as a JSON
    if not content_type_header:
        print("Missing Content-Type header. Try to decode with JSON default.")
        content_type =  "application/json"
    else:
        content_type = content_type_header.lower()
    
    if content_type not in supported_request_formats:
        raise BadRequestError(f"Unsupported Content-Type: {content_type}. Must be one of {supported_request_formats}")
    
    # Decode request based on header
    if content_type == "application/json":
        try:
            print("Decoding request body as JSON.")
            return request.get_json()
        except Exception as e:
            raise BadRequestError(f"Could not parse JSON payload: {e}")
    
    raise BadRequestError(f"Unsupported Content-Type: {content_type}. Must be one of {supported_request_formats}")
    

def encode_response(payload, status_code=200, isError=False, supported_response_formats=None, predictor_name="UnknownPredictor"):
    """
    Encodes the outgoing response payload based on the Accept header and supported response formats.
    Errors are ALWAYS sent as JSON.
    Prediction responses JSON only.

    Args:
        payload (dict): The Python dictionary to encode
        status_code (int, optional): The HTTP status code for the response. Defaults to 200.
        isError (bool, optional): Flag indicating if this is an error response. Defaults to False.
        supported_response_formats (list, optional): Lowercase list of supported response mime types. Defaults to None.
        predictor_name (str, optional): Name of Predictor. Defaults to "UnknownPredictor".
    """
    if supported_response_formats is None:
        supported_response_formats = ["application/json"] # Default
        
    if 'predictor_name' not in payload:
        payload = {"predictor_name": predictor_name, **payload}
        
    response_format = "application/json"

    try:
        return jsonify(payload), status_code
    except Exception as e:
            print("ERROR: Failed to serialize response as JSON.")
            raise ServerError(f"Internal Server Error: Failed to serialize response as JSON: {e}")