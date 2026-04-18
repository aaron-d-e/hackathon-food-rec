from flask_cors import cross_origin
from flask import request, render_template, redirect, make_response, jsonify

from .modules import infer_food_detection, detection_result_to_payload
from .utils import (
    process_webcam_capture,
    process_url_input,
    process_image_file,
    process_output_file,
    process_upload_file,
)


def _parse_bool(val, default=False):
    if val is None:
        return default
    return str(val).strip().lower() in ('1', 'true', 'yes', 'on')


def set_routes(app):
    @app.route('/')
    def homepage():
        resp = make_response(render_template("upload-file.html"))
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp


    @app.route('/url')
    def detect_by_url_page():
        return render_template("input-url.html")


    @app.route('/webcam')
    def detect_by_webcam_page():
        return render_template("webcam-capture.html")


    @app.route('/api/analyze', methods=['POST'])
    @cross_origin(origins='*')
    def api_analyze():
        """
        Multipart POST with field ``file`` (png/jpg/jpeg).
        Query params: model (default yolov8s), conf, iou, enhanced, ensemble, tta (all optional).
        Returns JSON: menu_items, ingredients, detections (no nutrition fields).
        """
        if 'file' not in request.files or not request.files['file'].filename:
            return jsonify(success=False, error='missing multipart field "file"'), 400

        filename, filepath, filetype = process_upload_file(request)

        if filetype != 'image':
            return jsonify(
                success=False,
                error='unsupported file type; use png, jpg, or jpeg',
            ), 400

        model = request.args.get('model', request.args.get('model_types', 'yolov8s'))
        model = str(model).lower()

        try:
            min_conf = float(request.args.get('conf', request.args.get('confidence', '0.15')))
            min_iou = float(request.args.get('iou', request.args.get('iou_threshold', '0.5')))
        except ValueError:
            return jsonify(success=False, error='invalid conf or iou (must be numbers)'), 400

        enhanced = _parse_bool(request.args.get('enhanced'))
        ensemble = _parse_bool(request.args.get('ensemble'))
        tta = _parse_bool(request.args.get('tta'))

        try:
            result_dict, _class_names, image_id, img_h, img_w, _ori = infer_food_detection(
                filepath,
                model_name=model,
                tta=tta,
                ensemble=ensemble,
                min_iou=min_iou,
                min_conf=min_conf,
                enhance_labels=enhanced,
            )
            payload = detection_result_to_payload(result_dict, image_id, img_w, img_h)
            payload['model'] = model
            return jsonify(payload)

        except FileNotFoundError as e:
            return jsonify(success=False, error=str(e)), 400
        except Exception as e:
            app.logger.exception('api/analyze failed')
            return jsonify(success=False, error=str(e)), 500

    @app.route('/analyze', methods=['POST', 'GET'])
    @cross_origin(supports_credentials=True)
    def analyze():
        if request.method == 'POST':
            out_name, filepath, filename, filetype, csv_name1, csv_name2 = None, None, None, None, None, None

            if 'webcam-button' in request.form:
                filename, filepath, filetype = process_webcam_capture(request)

            elif 'url-button' in request.form:
                filename, filepath, filetype = process_url_input(request)

            elif 'upload-button' in request.form:
                filename, filepath, filetype = process_upload_file(request)

            # Get all inputs in form
            min_iou = float(request.form.get('threshold-range')) / 100
            min_conf = float(request.form.get('confidence-range')) / 100
            model_types = request.form.get('model-types').lower()
            enhanced = request.form.get('enhanced') == 'on'
            ensemble = request.form.get('ensemble') == 'on'
            tta = request.form.get('tta') == 'on'
            segmentation = request.form.get('seg') == 'on'

            if filetype == 'image':
                out_name, output_path, output_type = process_image_file(filename, filepath, model_types, tta, ensemble, min_conf, min_iou, enhanced, segmentation)
            else:
                return render_template('detect-input-url.html', error_msg="Invalid input url!!!")

            filename, csv_name1, csv_name2 = process_output_file(output_path)

            if 'url-button' in request.form:
                return render_template('detect-input-url.html', out_name=out_name, segname=output_path, fname=filename, output_type=output_type, filetype=filetype, csv_name=csv_name1, csv_name2=csv_name2)

            elif 'webcam-button' in request.form:
                return render_template('detect-webcam-capture.html', out_name=out_name, segname=output_path, fname=filename, output_type=output_type, filetype=filetype, csv_name=csv_name1, csv_name2=csv_name2)

            return render_template('detect-upload-file.html', out_name=out_name, segname=output_path, fname=filename, output_type=output_type, filetype=filetype, csv_name=csv_name1, csv_name2=csv_name2)

        return redirect('/')


    @app.after_request
    def add_header(response):
        # Include cookie for every request
        response.headers.add('Access-Control-Allow-Credentials', True)

        # Prevent the client from caching the response
        if 'Cache-Control' not in response.headers:
            response.headers['Cache-Control'] = 'public, no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '-1'
        return response