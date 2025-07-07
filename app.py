import os
import csv
import re
import logging
from typing import List, Dict, Tuple, Optional
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles 
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from jinja2 import Template
import io

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Initialize FastAPI
app = FastAPI()

# Serve static files from the assets directory
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# Email regex pattern for validation
email_regex = re.compile(r"[^@]+@[^@]+\.[^@]+")

# Global SMTP configuration variable
smtp_config = {}

# CSV validation constants
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
MAX_ROWS = 10000  # Maximum number of rows
REQUIRED_COLUMNS = ['Email']  # Required columns
OPTIONAL_COLUMNS = ['Name', 'FirstName', 'LastName', 'Company', 'Title', 'Phone']  # Common optional columns

class CSVValidationError(Exception):
    """Custom exception for CSV validation errors"""
    pass

# Helper function to validate email addresses
def is_valid_email(email: str) -> bool:
    return re.fullmatch(email_regex, email) is not None

def validate_csv_file(file: UploadFile) -> None:
    """Validate CSV file before processing"""
    # Check file extension
    if not file.filename.lower().endswith('.csv'):
        raise CSVValidationError("File must be a CSV file")
    
    # Check file size
    if hasattr(file, 'size') and file.size > MAX_FILE_SIZE:
        raise CSVValidationError(f"File size exceeds maximum limit of {MAX_FILE_SIZE // (1024*1024)}MB")
    
    # Check if file is empty
    if file.filename == '':
        raise CSVValidationError("No file uploaded")

def validate_csv_content(content: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Validate and parse CSV content"""
    try:
        # Decode content
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        
        # Split into lines and remove empty lines
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        if len(lines) < 2:
            raise CSVValidationError("CSV must have at least a header row and one data row")
        
        if len(lines) > MAX_ROWS + 1:  # +1 for header
            raise CSVValidationError(f"CSV cannot have more than {MAX_ROWS} data rows")
        
        # Parse header
        header_line = lines[0]
        headers = [col.strip().strip('"') for col in header_line.split(',')]
        
        # Validate headers
        if not headers:
            raise CSVValidationError("CSV header is empty")
        
        # Check for required columns
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in headers]
        if missing_columns:
            raise CSVValidationError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Parse data rows
        data_rows = []
        for i, line in enumerate(lines[1:], 2):  # Start from line 2 (after header)
            try:
                # Simple CSV parsing (handles basic cases)
                values = []
                current_value = ""
                in_quotes = False
                
                for char in line:
                    if char == '"':
                        in_quotes = not in_quotes
                    elif char == ',' and not in_quotes:
                        values.append(current_value.strip())
                        current_value = ""
                    else:
                        current_value += char
                
                values.append(current_value.strip())  # Add last value
                
                # Validate row has correct number of columns
                if len(values) != len(headers):
                    raise CSVValidationError(f"Row {i}: Expected {len(headers)} columns, got {len(values)}")
                
                # Create row dictionary
                row = {}
                for header, value in zip(headers, values):
                    row[header] = value.strip('"')
                
                data_rows.append(row)
                
            except Exception as e:
                raise CSVValidationError(f"Error parsing row {i}: {str(e)}")
        
        if not data_rows:
            raise CSVValidationError("No valid data rows found in CSV")
        
        return headers, data_rows
        
    except UnicodeDecodeError:
        raise CSVValidationError("CSV file must be encoded in UTF-8")
    except Exception as e:
        if isinstance(e, CSVValidationError):
            raise e
        raise CSVValidationError(f"Error parsing CSV: {str(e)}")

def validate_email_data(data_rows: List[Dict[str, str]]) -> Tuple[List[str], List[str], List[Dict[str, str]]]:
    """Validate email data and return valid/invalid emails and valid rows"""
    valid_emails = []
    invalid_emails = []
    valid_rows = []
    
    for i, row in enumerate(data_rows, 1):
        email = row.get('Email', '').strip()
        
        if not email:
            invalid_emails.append(f"Row {i}: Empty email")
            continue
        
        if not is_valid_email(email):
            invalid_emails.append(f"Row {i}: Invalid email format - {email}")
            continue
        
        # Check for duplicate emails
        if email in valid_emails:
            invalid_emails.append(f"Row {i}: Duplicate email - {email}")
            continue
        
        valid_emails.append(email)
        valid_rows.append(row)
    
    return valid_emails, invalid_emails, valid_rows

def validate_template_variables(html_content: str, headers: List[str]) -> List[str]:
    """Validate that template variables exist in CSV headers"""
    # Extract template variables (anything between {{ }})
    template_vars = re.findall(r'\{\{(\w+)\}\}', html_content)
    
    missing_vars = []
    for var in template_vars:
        if var not in headers:
            missing_vars.append(var)
    
    return missing_vars

# Route for the index page
@app.get("/", response_class=HTMLResponse)
async def index():
    with open('Frontend/index.html', 'r') as f:
        return HTMLResponse(content=f.read(), status_code=200)

# Route to configure SMTP settings
@app.post("/configure_smtp")
async def configure_smtp(smtpHost: str = Form(...), smtpPort: int = Form(...),
                         smtpUser: str = Form(...), smtpPass: str = Form(...)):
    global smtp_config
    smtp_config = {
        'MAIL_SERVER': smtpHost,
        'MAIL_PORT': smtpPort,
        'MAIL_USERNAME': smtpUser,
        'MAIL_PASSWORD': smtpPass,
        'MAIL_STARTTLS': True,
        'MAIL_SSL_TLS': False,
        'USE_CREDENTIALS': True
    }

    logging.info(f"SMTP Config: {smtp_config}")

    return JSONResponse(content={'success': True, 'message': 'SMTP configuration updated successfully!'})

# Route to validate CSV file
@app.post("/validate_csv")
async def validate_csv(csvFile: UploadFile = File(...)):
    """Endpoint to validate CSV file before sending emails"""
    try:
        # Validate file
        validate_csv_file(csvFile)
        
        # Read and validate content
        content = await csvFile.read()
        headers, data_rows = validate_csv_content(content)
        
        # Validate email data
        valid_emails, invalid_emails, valid_rows = validate_email_data(data_rows)
        
        # Check template variables if HTML content is provided
        template_vars = []
        if 'htmlContent' in content:
            template_vars = validate_template_variables(content['htmlContent'], headers)
        
        return JSONResponse(content={
            'success': True,
            'message': f'CSV validation successful',
            'data': {
                'total_rows': len(data_rows),
                'valid_emails': len(valid_emails),
                'invalid_emails': len(invalid_emails),
                'headers': headers,
                'invalid_email_details': invalid_emails,
                'missing_template_vars': template_vars
            }
        })
        
    except CSVValidationError as e:
        return JSONResponse(
            content={'success': False, 'message': str(e)},
            status_code=400
        )
    except Exception as e:
        logging.error(f"Unexpected error during CSV validation: {e}")
        return JSONResponse(
            content={'success': False, 'message': 'Unexpected error during validation'},
            status_code=500
        )

# Route to send emails
@app.post("/send_emails")
async def send_emails(subject: str = Form(...), senderName: str = Form(...),
                      htmlContent: str = Form(...), csvFile: UploadFile = File(...)):
    if not smtp_config:
        raise HTTPException(status_code=400, detail="SMTP configuration is missing")

    try:
        # Validate CSV file
        validate_csv_file(csvFile)
        
        # Read and validate content
        content = await csvFile.read()
        headers, data_rows = validate_csv_content(content)
        
        # Validate email data
        valid_emails, invalid_emails, valid_rows = validate_email_data(data_rows)
        
        # Validate template variables
        missing_vars = validate_template_variables(htmlContent, headers)
        if missing_vars:
            raise HTTPException(
                status_code=400, 
                detail=f"Template variables not found in CSV: {', '.join(missing_vars)}"
            )
        
        if not valid_rows:
            raise HTTPException(
                status_code=400, 
                detail="No valid email addresses found in CSV"
            )

        # Initialize FastMail with the current SMTP configuration
        conf = ConnectionConfig(
            MAIL_USERNAME=smtp_config['MAIL_USERNAME'],
            MAIL_PASSWORD=smtp_config['MAIL_PASSWORD'],
            MAIL_FROM=smtp_config['MAIL_USERNAME'],
            MAIL_PORT=smtp_config['MAIL_PORT'],
            MAIL_SERVER=smtp_config['MAIL_SERVER'],
            MAIL_STARTTLS=smtp_config['MAIL_STARTTLS'],
            MAIL_SSL_TLS=smtp_config['MAIL_SSL_TLS'],
            USE_CREDENTIALS=smtp_config['USE_CREDENTIALS'],
            MAIL_FROM_NAME=senderName,
        )
        mail = FastMail(conf)

        # Send emails
        success_emails = []
        failed_emails = []
        
        for row in valid_rows:
            try:
                # Render HTML content with personalized data
                template = Template(htmlContent)
                personalized_html = template.render(row)

                # Prepare the email message
                message = MessageSchema(
                    subject=subject,
                    recipients=[row['Email']],
                    body=personalized_html,
                    subtype="html"
                )

                # Send the email
                await mail.send_message(message)
                success_emails.append(row['Email'])
                
            except Exception as e:
                logging.error(f"Failed to send email to {row['Email']}: {e}")
                failed_emails.append(row['Email'])

        # Prepare response
        response_data = {
            'success': True,
            'message': f'Email sending completed',
            'data': {
                'total_processed': len(valid_rows),
                'successful_sends': len(success_emails),
                'failed_sends': len(failed_emails),
                'invalid_emails': len(invalid_emails),
                'successful_emails': success_emails,
                'failed_emails': failed_emails,
                'invalid_email_details': invalid_emails
            }
        }

        return JSONResponse(content=response_data)

    except CSVValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error sending emails: {e}")
        raise HTTPException(status_code=500, detail=f"Error sending emails: {str(e)}")

# Vercel-specific function handler (for deployment)
@app.get("/vercel")
async def vercel():
    return JSONResponse(content={"message": "FastAPI is running on Vercel!"})

@app.get("/csv_template")
async def csv_template():
    """Downloadable CSV template for users"""
    output = io.StringIO()
    writer = csv.writer(output)
    # Add required and some optional columns
    writer.writerow(['Email', 'Name', 'Company'])
    writer.writerow(['example@email.com', 'John Doe', 'Acme Inc.'])
    output.seek(0)
    return StreamingResponse(
        output,
        media_type='text/csv',
        headers={
            'Content-Disposition': 'attachment; filename="mass_mailer_template.csv"'
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
