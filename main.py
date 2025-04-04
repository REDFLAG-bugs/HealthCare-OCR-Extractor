import os
import json
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tempfile
import pdf2image
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir() 
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

ocr = PaddleOCR(use_angle_cls=True, lang='en', version='PP-OCRv4')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Module 1: PDF Validation and Conversion
def validate_and_convert_pdf(pdf_path):
    if not pdf_path.lower().endswith('.pdf'):
        raise ValueError("Input file must be a '.pdf' file.")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File {pdf_path} does not exist.")

    try:
        images = pdf2image.convert_from_path(pdf_path)
        return images
    except Exception as e:
        logger.error(f"Error converting PDF to images: {str(e)}")
        raise ValueError(f"Failed to convert PDF: {str(e)}")
def extract_text_from_image(image, ocr):
    image_np = np.array(image)
    try:
        result = ocr.ocr(image_np, cls=True)
        if not result or not result[0]:
            return []

        regions = []
        for line in result[0]:
            bbox = line[0]
            text = line[1][0]  
            confidence = line[1][1]
            region_type = 'title' if confidence > 0.95 and len(text.split()) < 10 else 'text'
            regions.append({
                'type': region_type,
                'text': text,
                'bbox': bbox
            })

        return regions
    except Exception as e:
        logger.error(f"Error in OCR processing: {str(e)}")
        return []
def organize_into_sections(regions_list):
    sections = []
    current_section = None

    for region in regions_list:
        if region['type'] == 'title':
            if current_section is not None:
                sections.append(current_section)
            current_section = {
                'title': region['text'],
                'text_blocks': [],
                'pages': {region.get('page', 1)}
            }
        elif region['type'] == 'text' and current_section is not None:
            current_section['text_blocks'].append(region['text'])
            current_section['pages'].add(region.get('page', 1))
        elif region['type'] == 'text' and current_section is None:
            current_section = {
                'title': 'Unnamed Section',
                'text_blocks': [region['text']],
                'pages': {region.get('page', 1)}
            }

    if current_section is not None:
        sections.append(current_section)

    return sections
def chunk_sections(sections, num_pages, max_pages_per_chunk=6):
    if num_pages <= max_pages_per_chunk:
        full_text = '\n'.join([s['title'] + '\n' + '\n'.join(s['text_blocks']) for s in sections])
        return [full_text]

    chunks = []
    current_chunk = []
    current_pages = set()

    for section in sections:
        section_pages = section['pages']
        new_pages = current_pages.union(section_pages)

        if len(new_pages) > max_pages_per_chunk and current_chunk:
            chunk_text = '\n'.join([s['title'] + '\n' + '\n'.join(s['text_blocks']) for s in current_chunk])
            chunks.append(chunk_text)
            current_chunk = [section]
            current_pages = section_pages
        else:
            current_chunk.append(section)
            current_pages = new_pages

    if current_chunk:
        chunk_text = '\n'.join([s['title'] + '\n' + '\n'.join(s['text_blocks']) for s in current_chunk])
        chunks.append(chunk_text)

    return chunks

# Main Processing Function
def process_pdf(pdf_path):
    try:
        images = validate_and_convert_pdf(pdf_path)
        num_pages = len(images)
        all_regions = []
        for page_num, image in enumerate(images, start=1):
            regions = extract_text_from_image(image, ocr)
            for region in regions:
                region['page'] = page_num
                all_regions.append(region)
        sections = organize_into_sections(all_regions)
        chunks = chunk_sections(sections, num_pages)
        prescription_data = {
            'total_pages': num_pages,
            'chunks': chunks,
            'raw_text': '\n'.join(chunks)
        }

        return {
            'status': 'success',
            'message': f'Successfully processed {num_pages} page(s)',
            'data': prescription_data
        }
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return {
            'status': 'error',
            'message': str(e),
            'data': None
        }

@app.route('/api/health', methods=['GET'])
def health_check():
    """API endpoint to check service health."""
    return jsonify({
        'status': 'ok',
        'message': 'Service is running'
    })

@app.route('/api/extract_prescription', methods=['POST'])
def extract_prescription():
    if 'file' not in request.files:
        logger.warning("No file part in the request")
        return jsonify({
            'status': 'error',
            'message': 'No file part in the request'
        }), 400

    file = request.files['file']
    if file.filename == '':
        logger.warning("No file selected")
        return jsonify({
            'status': 'error',
            'message': 'No file selected'
        }), 400
    if not allowed_file(file.filename):
        logger.warning(f"File type not allowed: {file.filename}")
        return jsonify({
            'status': 'error',
            'message': 'File type not allowed. Please upload a PDF file.'
        }), 400

    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        logger.info(f"Processing file: {filename}")
        result = process_pdf(file_path)
        os.remove(file_path)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to process the PDF: {str(e)}'
        }), 500
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)