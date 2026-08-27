# How to Enable Geocoding API

## Step-by-Step Instructions

### 1. Go to Google Cloud Console
Visit: https://console.cloud.google.com/

### 2. Select Your Project
- If you don't have a project, create one
- Select the project that contains your Google Maps API key

### 3. Navigate to APIs & Services
- Click on the hamburger menu (☰) in the top left
- Go to **"APIs & Services"** → **"Library"**

### 4. Search for Geocoding API
- In the search bar, type: **"Geocoding API"**
- Click on **"Geocoding API"** from the results

### 5. Enable the API
- Click the **"ENABLE"** button
- Wait for the API to be enabled (usually takes a few seconds)

### 6. Verify API is Enabled
- You should see a green checkmark or "API enabled" message
- The page will show usage statistics and quota information

### 7. (Optional) Enable Places API for Better Search
For enhanced search suggestions, also enable:
- Search for **"Places API"**
- Click **"ENABLE"**

## Required APIs Checklist

Make sure these APIs are enabled in your Google Cloud project:

✅ **Maps JavaScript API** - For displaying the map
✅ **Geocoding API** - For location search (REQUIRED for search feature)
✅ **Drawing Library** - For geofence drawing (included with Maps JavaScript API)
✅ **Geometry Library** - For geofence calculations (included with Maps JavaScript API)
✅ **Places API** - Optional, for better search suggestions

## Verify Your API Key Has Access

1. Go to **"APIs & Services"** → **"Credentials"**
2. Click on your API key
3. Under **"API restrictions"**, make sure:
   - Either "Don't restrict key" is selected, OR
   - The following APIs are listed:
     - Maps JavaScript API
     - Geocoding API
     - Places API (if enabled)

## Testing the Geocoding API

After enabling, test the search feature:

1. Open your application
2. Type a location in the search box (e.g., "Accra, Ghana")
3. Click search or press Enter
4. The map should center on the location

## Common Issues

### "Geocoding API request denied"
- **Solution**: Make sure Geocoding API is enabled in your project
- Check that your API key has access to Geocoding API

### "OVER_QUERY_LIMIT"
- **Solution**: You've exceeded the free quota (usually 40,000 requests/month)
- Check your usage in Google Cloud Console
- Consider upgrading your plan if needed

### "ZERO_RESULTS"
- **Solution**: The location you searched for doesn't exist or is misspelled
- Try a more specific location name

## API Quotas (Free Tier)

- **Geocoding API**: 40,000 requests per month (free)
- **Places API**: Different pricing, check Google Cloud Console

## Need Help?

- Google Cloud Console: https://console.cloud.google.com/
- Geocoding API Documentation: https://developers.google.com/maps/documentation/geocoding
- API Status: https://status.cloud.google.com/

