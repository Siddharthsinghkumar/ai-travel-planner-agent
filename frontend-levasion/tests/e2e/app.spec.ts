import { test, expect } from '@playwright/test';

test.describe('L\'ÉVASION End-to-End Flow', () => {
  test('User plans trip via LLM and holds flight', async ({ page }) => {
    // Navigate to the Explore page
    await page.goto('http://localhost:5173/');

    // Ensure we are on the Explore page (or whatever the main CTA is based on the minimalist single omni-box design)
    const searchBox = page.locator('input[type="text"]').first();
    await searchBox.waitFor({ state: 'visible' });

    // Type a natural language prompt
    await searchBox.fill('I want to fly to Paris next week from NYC in business class');
    await searchBox.press('Enter');

    // Should redirect to Planner with SSE stream
    await expect(page).toHaveURL(/.*\/planner/, { timeout: 10000 });

    // Click "Generate Itinerary"
    const generateBtn = page.locator('button', { hasText: 'Generate Itinerary' });
    await generateBtn.click();

    // Wait for the stream to complete and the flight card to appear.
    const holdButton = page.locator('button', { hasText: /Hold|Book|Select/i }).first();
    await expect(holdButton).toBeVisible({ timeout: 25000 });

    // Click the button to hold/book the flight
    await holdButton.click();

    // Should redirect to Checkout or Profile depending on the flow
    await expect(page).toHaveURL(/.*\/(checkout|profile)/, { timeout: 10000 });
  });
});
