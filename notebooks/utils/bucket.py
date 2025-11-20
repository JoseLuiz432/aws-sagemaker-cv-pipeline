import os

def create_bucket_if_not_exists(s3_client, bucket_name, region):
    """
    Cria um bucket S3 se ele não existir.
    Args:
        s3_client: Cliente boto3 S3.
        bucket_name: Nome do bucket a ser criado.
        region: Região onde o bucket será criado.
    Returns:
        None
    """
    try:
        if region == 'us-east-1':
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region}
            )
        print(f"✅ Bucket '{bucket_name}' criado/verificado com sucesso.")
    except Exception as e:
        print(f"❌ Erro ao criar bucket: {e}")
        
        
def upload_to_s3(session, local_path, bucket, s3_prefix):
    """
    Faz upload de dados locais para um bucket S3.
    Args:
        session: Sessão do SageMaker.
        local_path: Caminho local dos dados a serem enviados.
        bucket: Nome do bucket S3.
        s3_prefix: Prefixo (pasta) no bucket S3 onde os dados serão armazenados.
    Returns:
        s3_uri: URI do S3 onde os dados foram enviados.
    """
    
    if os.name == 'nt':
        return upload_to_s3_windows(session, local_path, bucket, s3_prefix)
    
    print(f"Iniciando upload de {local_path} para s3://{bucket}/{s3_prefix}")
    
    s3_uri = session.upload_data(
        path=local_path, 
        bucket=bucket, 
        key_prefix=s3_prefix
    )
    return s3_uri

def upload_to_s3_windows(session, local_path, bucket, s3_prefix):
    """
    Faz upload corrigindo automaticamente barras invertidas (Windows) para S3.
    """
    print(f"Iniciando upload de '{local_path}' para 's3://{bucket}/{s3_prefix}'")
    
    s3_client = session.boto_session.client('s3')
    
    count = 0
    for root, dirs, files in os.walk(local_path):
        for file in files:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, local_path)
            s3_key_path = os.path.join(s3_prefix, relative_path).replace("\\", "/")
            s3_client.upload_file(full_path, bucket, s3_key_path)
            count += 1
            
            if count % 50 == 0: 
                print(f"  Upload: {s3_key_path}")

    print(f"Upload finalizado! {count} arquivos enviados.")
    return f"s3://{bucket}/{s3_prefix}"