# Define the UDP port number
$Port = 10514

# Create a UDP endpoint
$Endpoint = New-Object System.Net.IPEndPoint ([System.Net.IPAddress]::Any, $Port)

# Create a UDP client
$UdpClient = New-Object System.Net.Sockets.UdpClient $Port

Write-Host "Listening on UDP port $Port..."

try {
    # Continuously listen for messages
    while ($true) {
        # Receive bytes
        $Bytes = $UdpClient.Receive([ref]$Endpoint)
        # Decode the message
        $Message = [System.Text.Encoding]::ASCII.GetString($Bytes)
        # Display the message
        Write-Host "Received message: $Message"
    }
}
catch {
    Write-Error "An error occurred: $_"
}
finally {
    # Clean up
    $UdpClient.Close()
    Write-Host "Listener stopped."
}